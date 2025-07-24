#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <optional>


// サムネイル選定やスコア処理用のスコア付き構造体（処理中間用）
// 1フレーム単位のデータを保持
struct FrameData {
    int frameIndex;                              // フレーム番号(デバッグ専用)
    std::vector<float> treatmentProbabilities;   // 推論スコア（15クラス）
	std::vector<int> sceneBinaryLabels;          // バイナリ化されたシーンラベル（0〜5の主クラス）
    int sceneLabel = -1;                         // 平滑化されたラベル
    float sceneProb = 0.0f;                      // シーンクラスの確率
    float eventProbsSum = 0.0f;                  // イベントクラスの確率の合計
};


// スライディングウィンドウ内のデータを管理するクラス
class FrameWindow {
private:
    int windowSize;
    int halfWindow;

    std::deque<cv::Mat> imageBuffer;
    std::deque<FrameData> dataBuffer;

public:
    FrameWindow(int windowSize);

	// ウィンドウ内にフレームを追加
    void push(const cv::Mat& image, const FrameData& data);

    // 中心フレームの取得
    const cv::Mat& getCenterImage() const;
    const FrameData& getCenterData() const;

    // 最新フレームのインデックス（バッファ先頭）を元に返す
    int getCenterFrameIndex() const;

    // 読み取り専用の参照を返す
    const std::deque<FrameData>& getDataBuffer() const {
        return dataBuffer;
    }

    // バッファの中心位置（windowSize / 2）
    int getWindowOffset() const;

    // ウィンドウ中心のシーンラベルを設定
    void setCenterSceneLabel(int centerLabel);

    // ウィンドウ内のフレームのシーンラベルを取得
    std::vector<std::vector<int>> getSceneLabels() const {
        std::vector<std::vector<int>> labels;
        for (const auto& data : dataBuffer) {
            labels.push_back(data.sceneBinaryLabels);
        }
        return labels;
    }

    // バッファが満たされているか
    bool isReady() const;

	// バッファのサイズ(debug)
    int size() const;

	// 中心フレームのframeDataの中身を出力
    void logFirstFrameData() const {
        if (!dataBuffer.empty()) {
            const FrameData& firstFrame = dataBuffer.front();
            std::cout << "First Frame Index: " << firstFrame.frameIndex
                      << ", Scene Label: " << firstFrame.sceneLabel
                      << ", Scene Prob: " << firstFrame.sceneProb
                      << ", Event Probs Sum: " << firstFrame.eventProbsSum << std::endl;
        }
	}

	// デバッグ用：ウィンドウ内のインデックスを表示
    void logWindowIndices() const {
        std::cout << "Window Indices: ";
        for (const auto& data : dataBuffer) {
            std::cout << data.frameIndex << " ";
        }
        std::cout << std::endl;
	}
};


std::optional<int> processSceneLabelSlidingWindow(
    const std::deque<std::vector<int>>& windowSceneLabelBuffer,
    int prevSceneLabel
);


// スライディングウィンドウによるラベル決定を行うクラス
class SceneLabelSmoother {
public:
    // コンストラクタ：ラベル数とウィンドウサイズを指定
    SceneLabelSmoother(int numSceneLabels, int windowSize);

    // 新しいフレームを追加して、中心ラベルを返す（変化なし/null時は nullopt）
    std::optional<int> processSceneLabel(const std::vector<std::vector<int>>& windowSceneLabels);

    // 中心フレームのインデックス取得のためのオフセット（windowSize / 2）
    int getWindowOffset() const;

private:
    int numSceneLabels;
    int windowSize;
    int halfWindow;
    int prevLabel = -1;

    std::deque<std::vector<int>> windowBuffer; // ラベルの履歴
};