#include "thumbnail_selector.h"
#include "debug.h"
#include <algorithm>
#include <set>

SceneSegmentManager::SceneSegmentManager(int topK, int frameGap)
    :topK(topK), frameGap(frameGap) {
}

float SceneSegmentManager::computeDeeplearningScore(float sceneProb, float eventProbsSum) {
    // 深層学習スコアの計算式
    return sceneProb - eventProbsSum;
}

float SceneSegmentManager::computeHighFrequencyEnergy(const cv::Mat& image) {
    // 高周波エネルギーの計算
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    }
    else {
        gray = image.clone();
    }

    gray.convertTo(gray, CV_32F);

    // DFT（複素数出力）
    cv::Mat dftImg;
    cv::dft(gray, dftImg, cv::DFT_COMPLEX_OUTPUT);

    // FFTShift
    int cx = dftImg.cols / 2;
    int cy = dftImg.rows / 2;
    cv::Mat q0(dftImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(dftImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(dftImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(dftImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // 高周波マスク作成（低周波中心を0に）
    int radius = std::min(gray.rows, gray.cols) / 8;
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
    cv::circle(mask, cv::Point(cx, cy), radius, cv::Scalar(0), -1);

    // --- マスク（中心を四角形でカット） ---
    //int radius = std::min(gray.rows, gray.cols) / 8;
    //cv::Mat mask = cv::Mat::ones(gray.rows, gray.cols, CV_32F);
    //cv::Rect lowFreqBox(cx - radius, cy - radius, radius * 2, radius * 2);
    //mask(lowFreqBox) = 0.0f;  // 中心の低周波領域を0にする（四角形）

    // エネルギー計算
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

    return totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0.0;
}


void SceneSegmentManager::update(const FrameData& data, const cv::Mat& image) {
    // 
    int label = data.sceneLabel;  // シーンラベルを取得
	float dlScore = computeDeeplearningScore(data.sceneProb, data.eventProbsSum);  // 深層学習スコア
	float hfScore = computeHighFrequencyEnergy(image.clone());  // 高周波エネルギー

    // 
    ThumbnailCandidate candidate{ data.frameIndex, image.clone(), dlScore, hfScore };

    // ラベルが変わった場合、前のラベルのセグメントを確定
    if (prevLabel != -1 && label != prevLabel) {
        auto& prevSeg = currentSegment[prevLabel];
        if (prevSeg.length > longestSegment[prevLabel].length) {
            longestSegment[prevLabel] = prevSeg;
        }
        currentSegment.erase(prevLabel);
    }

    // セグメント更新
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

void SceneSegmentManager::finalize() {
    for (const auto& [label, seg] : currentSegment) {
        if (seg.length > longestSegment[label].length) {
            longestSegment[label] = seg;
        }
    }
    currentSegment.clear();
}

std::map<int, std::vector<ThumbnailCandidate>> SceneSegmentManager::getFinalThumbnails() const {
    std::map<int, std::vector<ThumbnailCandidate>> result;
    for (const auto& [label, seg] : longestSegment) {
        result[label] = selectWithFrameGap(seg.topKThumbnails, frameGap, topK);
    }
    return result;
}

std::vector<ThumbnailCandidate> SceneSegmentManager::selectWithFrameGap(
    std::priority_queue<ThumbnailCandidate> candidates,
    int frameGap, int topK) const {

    std::vector<ThumbnailCandidate> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
    }

    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.combinedScore() > b.combinedScore();
        });

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

void SceneSegmentManager::logSummary() const {
    log("=== ラベルごとの最長区間 ===", true);
    for (const auto& [label, seg] : longestSegment) {
        log("Label " + std::to_string(label)
            + " [" + std::to_string(seg.startFrameIndex) + "," + std::to_string(seg.endFrameIndex) + "]"
            + " (length=" + std::to_string(seg.length) + ")", true);

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

        // 画像保存もできる
        cv::imwrite(savePath + "_label_" + std::to_string(label) + ".png", tile);
    }
}