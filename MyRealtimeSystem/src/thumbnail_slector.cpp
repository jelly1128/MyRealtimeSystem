#include "thumbnail_selector.h"
#include "debug.h"
#include <algorithm>
#include <set>


// ==================== SceneSegmentManager 実装 ====================

/**
 * @brief コンストラクタ
 * @param topK サムネイル候補数
 * @param frameGap サムネイルの最小フレーム間隔
 */
SceneSegmentManager::SceneSegmentManager(int topK, int frameGap)
    :topK(topK), frameGap(frameGap) {
}


/**
 * @brief 深層学習スコアを計算
 * @param sceneProb シーンの確率
 * @param eventProbsSum イベントクラスの確率の合計
 * @return 深層学習スコア
 */
float SceneSegmentManager::computeDeepLearningScore(float sceneProb, float eventProbsSum) {
    // 深層学習スコアの計算式
    return sceneProb - eventProbsSum;
}


/**
 * @brief 高周波エネルギーを計算
 * @param image 入力画像
 * @return 高周波エネルギーの比率
 */
float SceneSegmentManager::computeHighFrequencyEnergy(const cv::Mat& image) {
    // 1. グレースケール変換
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    }
    else {
        gray = image.clone();
    }
    gray.convertTo(gray, CV_32F);

    // 2. DFT（複素数出力）
    cv::Mat dftImg;
    cv::dft(gray, dftImg, cv::DFT_COMPLEX_OUTPUT);

    // 3. FFTShift（低周波成分を中央に）
    int cx = dftImg.cols / 2;
    int cy = dftImg.rows / 2;
    cv::Mat q0(dftImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(dftImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(dftImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(dftImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // 4. 高周波マスク作成（低周波中心を0に）
    int radius = std::min(gray.rows, gray.cols) / 8;
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
    cv::circle(mask, cv::Point(cx, cy), radius, cv::Scalar(0), -1);

    // 4. 高周波マスク作成（中心を四角形でカット）
    //int radius = std::min(gray.rows, gray.cols) / 8;
    //cv::Mat mask = cv::Mat::ones(gray.rows, gray.cols, CV_32F);
    //cv::Rect lowFreqBox(cx - radius, cy - radius, radius * 2, radius * 2);
    //mask(lowFreqBox) = 0.0f;  // 中心の低周波領域を0にする（四角形）

    // 5. エネルギー計算
    std::vector<cv::Mat> channels(2);
    cv::split(dftImg, channels);
    cv::Mat mag;
    cv::magnitude(channels[0], channels[1], mag);
    cv::pow(mag, 2, mag);  // エネルギー = 振幅^2

    double totalEnergy = cv::sum(mag)[0];

    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F);
    cv::Mat highFreqOnly;
    cv::multiply(mag, maskFloat, highFreqOnly);
    double highFreqEnergy = cv::sum(highFreqOnly)[0];

    // 6. 正規化して返す
    return totalEnergy > 0 ? static_cast<float>(highFreqEnergy / totalEnergy) : 0.0f;
}


/**
 * @brief 1フレーム分（スライディングウィンドウの中心フレーム）のサムネイル候補を管理・更新
 * @param data フレームデータ
 * @param image 入力画像
 */
void SceneSegmentManager::update(const FrameData& data, const cv::Mat& image) {
	// １フレーム分のデータをサムネイル候補として登録
    int label = data.sceneLabel;
	float deepLearningScore = computeDeepLearningScore(data.sceneProb, data.eventProbsSum);
    float highFrequencyScore = computeHighFrequencyEnergy(image);

    ThumbnailCandidate candidate{ data.frameIndex, image.clone(), deepLearningScore, highFrequencyScore };

    // ラベルが変わった場合、前の区間を確定
    if (prevLabel != -1 && label != prevLabel) {
        auto& prevSeg = currentSegment[prevLabel];
        if (prevSeg.length > longestSegment[prevLabel].length) {
            longestSegment[prevLabel] = prevSeg;
        }
        currentSegment.erase(prevLabel);
    }

    // セグメントを更新
    auto& seg = currentSegment[label];
    if (seg.length == 0) {
        seg.startFrameIndex = data.frameIndex;
    }
    seg.endFrameIndex = data.frameIndex;
    seg.length++;
    seg.topKThumbnails.push(candidate);
    if (seg.topKThumbnails.size() > topK) seg.topKThumbnails.pop();

    prevLabel = label;
}


/**
 * @brief 最長区間を確定
 */
void SceneSegmentManager::finalize() {
    for (const auto& [label, seg] : currentSegment) {
        if (seg.length > longestSegment[label].length) {
            longestSegment[label] = seg;
        }
    }
    currentSegment.clear();
}


/**
 * @brief ラベルごとの最終Top-Kサムネイル候補を取得
 */
std::map<int, std::vector<ThumbnailCandidate>> SceneSegmentManager::getFinalThumbnails() const {
    std::map<int, std::vector<ThumbnailCandidate>> result;
    for (const auto& [label, seg] : longestSegment) {
        result[label] = selectWithFrameGap(seg.topKThumbnails, frameGap, topK);
    }
    return result;
}

/**
 * @brief フレーム間隔を考慮してTop-K候補を選定
 */
std::vector<ThumbnailCandidate> SceneSegmentManager::selectWithFrameGap(
    std::priority_queue<ThumbnailCandidate> candidates,
    int frameGap, int topK) const
{
    // priority_queue → vector（高スコア順）
    std::vector<ThumbnailCandidate> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.combinedScore() > b.combinedScore();
    });

	// フレーム間隔を考慮して選定
    std::vector<ThumbnailCandidate> selected;
    std::set<int> usedFrames;

    for (const auto& cand : sorted) {
        bool tooClose = false;
        for (int idx : usedFrames) {
            if (std::abs(cand.frameIndex - idx) < frameGap) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            selected.push_back(cand);
            usedFrames.insert(cand.frameIndex);
            if (selected.size() == topK) break;
        }
    }
    return selected;
}


/**
 * @brief ラベルごとの最長区間とサムネイル候補をログ出力
 */
void SceneSegmentManager::logSummary() const {
    log("=== ラベルごとの最長区間 ===", true);
    for (const auto& [label, seg] : longestSegment) {
        log("Label " + std::to_string(label)
            + " [" + std::to_string(seg.startFrameIndex) + "," + std::to_string(seg.endFrameIndex) + "]"
            + " (length=" + std::to_string(seg.length) + ")", true);

        // サムネイル候補のスコアを降順で出力
        std::priority_queue<ThumbnailCandidate> thumbs = seg.topKThumbnails;
        std::vector<ThumbnailCandidate> thumbsVec;
        while (!thumbs.empty()) {
            thumbsVec.push_back(thumbs.top());
            thumbs.pop();
        }
        std::sort(thumbsVec.begin(), thumbsVec.end(), [](const auto& a, const auto& b) {
            return a.combinedScore() > b.combinedScore();
            });
        for (const auto& cand : thumbsVec) {
            log("  Frame " + std::to_string(cand.frameIndex)
                + ", DLScore=" + std::to_string(cand.deepLearningScore)
                + ", HiFreq=" + std::to_string(cand.highFrequencyScore)
                + ", Score=" + std::to_string(cand.combinedScore()), true);
        }
    }
}

// ==================== サムネイル可視化ユーティリティ ====================

/**
 * @brief サムネイル画像をグリッド状に合成
 * @param thumbs      サムネイル画像配列
 * @param thumbWidth  サムネイル幅[pixel]
 * @param thumbHeight サムネイル高さ[pixel]
 * @param gridCols    グリッド列数（0で自動計算）
 * @param gridRows    グリッド行数（0で自動計算）
 * @param margin      サムネイル間マージン[pixel]
 * @param bgColor     背景色
 * @return 合成画像
 */
cv::Mat createThumbnailTile(
    const std::vector<cv::Mat>& thumbs,
    int thumbWidth = 160, int thumbHeight = 120,
    int gridCols = 0, int gridRows = 0,
    int margin = 8, cv::Scalar bgColor = cv::Scalar(240, 240, 240)
) {
    int numThumbs = thumbs.size();
    if (gridCols == 0) gridCols = std::ceil(std::sqrt(numThumbs));
    if (gridRows == 0) gridRows = std::ceil((float)numThumbs / gridCols);

    int outWidth = thumbWidth * gridCols + margin * (gridCols + 1);
    int outHeight = thumbHeight * gridRows + margin * (gridRows + 1);
    cv::Mat canvas(outHeight, outWidth, CV_8UC3, bgColor);

    for (int i = 0; i < numThumbs; ++i) {
        int row = i / gridCols;
        int col = i % gridCols;
        int x = margin + col * (thumbWidth + margin);
        int y = margin + row * (thumbHeight + margin);

        if (!thumbs[i].empty()) {
            cv::Mat thumbResized;
            cv::resize(thumbs[i], thumbResized, cv::Size(thumbWidth, thumbHeight));
            thumbResized.copyTo(canvas(cv::Rect(x, y, thumbWidth, thumbHeight)));
        }
    }
    return canvas;
}


/**
 * @brief ラベルごとのサムネイルを画像に合成し保存する
 * @param thumbsPerLabel ラベルごとのサムネイル配列
 * @param savePath       保存先パス（"_label_0.png"等が付与される）
 * @param thumbWidth     サムネイル幅[pixel]
 * @param thumbHeight    サムネイル高さ[pixel]
 * @param gridCols       グリッド列数
 */
void visualizeThumbnailsPerLabel(
    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
	const std::string& savePath,
    int thumbWidth, int thumbHeight, int gridCols
) {
    for (const auto& [label, thumbs] : thumbsPerLabel) {
        std::vector<cv::Mat> thumbImgs;
        for (const auto& cand : thumbs) {
            if (!cand.frame.empty()) {
                cv::Mat imgBGR;
                if (cand.frame.type() == CV_32FC3) {
                    cand.frame.convertTo(imgBGR, CV_8UC3, 255.0); // 0～1→0～255
                }
                else {
                    imgBGR = cand.frame.clone();
                }
                // 必要ならBGR変換
                cv::cvtColor(imgBGR, imgBGR, cv::COLOR_RGB2BGR);
                thumbImgs.push_back(imgBGR);
            }
        }
        if (thumbImgs.empty()) continue;

        cv::Mat tile = createThumbnailTile(thumbImgs, thumbWidth, thumbHeight, gridCols);

        // 画像表示
        /*std::string winName = "Label " + std::to_string(label) + " Thumbnails";
        cv::imshow(winName, tile);
        cv::waitKey(0);*/

        // 画像保存
        cv::imwrite(savePath + "_label_" + std::to_string(label) + ".png", tile);
    }
}


void logFinalThumbnails(const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel)
{
    log("===== ラベルごとの最終Top-Kサムネイル候補一覧 =====", true);
    for (const auto& [label, thumbs] : thumbsPerLabel) {
        log("ラベル: " + std::to_string(label) + "（件数: " + std::to_string(thumbs.size()) + "）", true);
        int i = 0;
        for (const auto& cand : thumbs) {
            log("  [" + std::to_string(i++) +
                "] フレーム番号: " + std::to_string(cand.frameIndex) +
                ", 深層学習スコア: " + std::to_string(cand.deepLearningScore) +
                ", 高周波スコア: " + std::to_string(cand.highFrequencyScore) +
                ", 合算スコア: " + std::to_string(cand.combinedScore()), true);
        }
    }
    log("===============================================", true);
}