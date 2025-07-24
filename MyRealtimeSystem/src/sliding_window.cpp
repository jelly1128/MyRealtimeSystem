#include "sliding_window.h"


// スライディングウィンドウを使用してシーンラベルを1つにまとめる関数
std::optional<int> processSceneLabelSlidingWindow(
    const std::deque<std::vector<int>>& windowSceneLabelBuffer,
	int prevSceneLabel
) {
	// シーンクラス数を取得
	int numSceneClasses = windowSceneLabelBuffer.front().size();
    std::vector<int> classCounts(numSceneClasses, 0);

    // 現ウィンドウ内で各クラスの合計値を計算
    for (int i = 0; i < windowSceneLabelBuffer.size(); ++i) {
        for (int c = 0; c < numSceneClasses; ++c) {
            classCounts[c] += windowSceneLabelBuffer[i][c];
        }
    }

    // 最多クラスを抽出
    int maxCount = *std::max_element(classCounts.begin(), classCounts.end());
    std::vector<int> maxIndices;
    for (int c = 0; c < numSceneClasses; ++c) {
        if (classCounts[c] == maxCount) maxIndices.push_back(c);
    }

    int decidedLabel;
    if (maxIndices.size() == 1) {
        decidedLabel = maxIndices[0];  // 明確な多数決
    }
    else if (prevSceneLabel != -1) {
        decidedLabel = prevSceneLabel; // 同点 → 前回のラベルを再利用
    }
    else {
        decidedLabel = 0;  // 初回など → 白色光
    }

    return decidedLabel;
}


// スライディングウィンドウ内のデータを管理するクラス
FrameWindow::FrameWindow(int windowSize)
    : windowSize(windowSize), halfWindow(windowSize / 2) {
}

void FrameWindow::push(const cv::Mat& image, const FrameData& data) {
    imageBuffer.push_back(image.clone());  // clone推奨：外側で画像が破棄される可能性に備える
    dataBuffer.push_back(data);

    if (imageBuffer.size() > windowSize) imageBuffer.pop_front();
    if (dataBuffer.size() > windowSize) dataBuffer.pop_front();
}

const cv::Mat& FrameWindow::getCenterImage() const {
    return imageBuffer[halfWindow];
}

const FrameData& FrameWindow::getCenterData() const {
    return dataBuffer[halfWindow];
}

int FrameWindow::getCenterFrameIndex() const {
    return dataBuffer[halfWindow].frameIndex;
}

int FrameWindow::getWindowOffset() const {
    return halfWindow;
}

void FrameWindow::setCenterSceneLabel(int centerLabel){
    if (!dataBuffer.empty()) {
		dataBuffer[halfWindow].sceneLabel = centerLabel;  // 中心フレームのシーンラベルを設定
		dataBuffer[halfWindow].sceneProb = dataBuffer[halfWindow].treatmentProbabilities[centerLabel]; // シーン確率を更新
    }
}

int FrameWindow::size() const {
    return imageBuffer.size();
}

bool FrameWindow::isReady() const {
    return imageBuffer.size() >= windowSize;
}


// スライディングウィンドウによるラベル決定を行うクラス
SceneLabelSmoother::SceneLabelSmoother(int numSceneLabels, int windowSize)
    : numSceneLabels(numSceneLabels), windowSize(windowSize), halfWindow(windowSize / 2) {
}


std::optional<int> SceneLabelSmoother::processSceneLabel(const std::vector<std::vector<int>>& windowSceneLabels) {
	// ウィンドウ内に十分なデータがない場合は何もしない
    if (windowSceneLabels.size() < windowSize) {
        return std::nullopt;  // ウィンドウが満たされていない
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

int SceneLabelSmoother::getWindowOffset() const {
    return halfWindow;
}
