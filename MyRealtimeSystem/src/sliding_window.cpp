#include "sliding_window.h"


// =============================
// @section FrameWindow 実装
// =============================

/**
 * @brief コンストラクタ
 * @param windowSize ウィンドウサイズ（フレーム数）
 */
FrameWindow::FrameWindow(int windowSize)
    : windowSize(windowSize), halfWindow(windowSize / 2) {}


/**
 * @brief ウィンドウに画像＋データを追加
 */
void FrameWindow::push(const cv::Mat& image, const FrameData& data) {
    windowedImages.push_back(image.clone());  // clone必須:外で画像が消える場合に備える
    windowedFrameData.push_back(data);
    if (windowedImages.size() > windowSize) windowedImages.pop_front();
    if (windowedFrameData.size() > windowSize) windowedFrameData.pop_front();
}


/**
 * @brief ウィンドウ中心のデータを返す
 */
const cv::Mat& FrameWindow::getCenterImage() const {
    return windowedImages[halfWindow];
}

const FrameData& FrameWindow::getCenterData() const {
    return windowedFrameData[halfWindow];
}


/**
 * @brief ウィンドウ中心フレームのsceneLabel・sceneProbを書き換える
 */
void FrameWindow::setCenterSceneLabel(int centerLabel) {
    if (!windowedFrameData.empty()) {
        windowedFrameData[halfWindow].sceneLabel = centerLabel;
        windowedFrameData[halfWindow].sceneProb = windowedFrameData[halfWindow].treatmentProbabilities[centerLabel];
    }
}


/**
 * @brief ウィンドウ内のsceneラベル（バイナリ配列）リストを返す
 */
std::vector<std::vector<int>> FrameWindow::getSceneLabels() const {
    std::vector<std::vector<int>> labels;
    for (const auto& data : windowedFrameData) {
        labels.push_back(data.sceneBinaryLabels);
    }
    return labels;
}


/**
 * @brief ウィンドウ内のフレームデータのログ出力
 */
void FrameWindow::logFirstFrameData() const {
    if (!windowedFrameData.empty()) {
        const FrameData& firstFrame = windowedFrameData.front();
        std::cout << "First Frame Index: " << firstFrame.frameIndex
            << ", Scene Label: " << firstFrame.sceneLabel
            << ", Scene Prob: " << firstFrame.sceneProb
            << ", Event Probs Sum: " << firstFrame.eventProbsSum << std::endl;
    }
}


// =============================
// @section SceneLabelSmoother 実装
// =============================

/**
 * @brief コンストラクタ
 * @param numSceneLabels シーンラベル数
 * @param windowSize     ウィンドウサイズ
 */
SceneLabelSmoother::SceneLabelSmoother(int numSceneLabels, int windowSize)
    : numSceneLabels(numSceneLabels), windowSize(windowSize), halfWindow(windowSize / 2) {
}


/**
 * @brief ウィンドウ内ラベルから中心ラベルを決定
 */
std::optional<int> SceneLabelSmoother::processSceneLabel(const std::vector<std::vector<int>>& windowSceneLabels) {
	// ウィンドウ内に十分なデータがない場合は何もしない
    if (windowSceneLabels.size() < windowSize) {
        return std::nullopt;
	}

	// ウィンドウ内のラベルを集計
    std::vector<int> classCounts(numSceneLabels, 0);
    for (const auto& labels : windowSceneLabels) {
        for (int i = 0; i < numSceneLabels; ++i) {
            classCounts[i] += labels[i];
        }
    }

	// 最も多いクラスを決定
    int maxCount = *std::max_element(classCounts.begin(), classCounts.end());
    std::vector<int> maxIndices;
    for (int i = 0; i < numSceneLabels; ++i) {
        if (classCounts[i] == maxCount) {
            maxIndices.push_back(i);
        }
    }

	// 決定されたラベルを選択
    int decidedLabel;
	if (maxIndices.size() == 1) {     // 明確な多数決
        decidedLabel = maxIndices[0];
    }
	else if (prevLabel != -1) {       // 同点の場合、前回のラベルを再利用
        decidedLabel = prevLabel;
    }
    else {
        decidedLabel = 0;             // 初期値（白色光など）を返す
    }

    prevLabel = decidedLabel;
    return decidedLabel;
}


/**
 * @brief ウィンドウのオフセット（中心フレームインデックス）を取得
 */
int SceneLabelSmoother::getWindowOffset() const {
    return halfWindow;
}
