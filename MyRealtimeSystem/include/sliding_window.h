#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <optional>


/// ==========================
/// @section フレーム情報構造体
/// ==========================

/**
 * @brief サムネイル選定やスコア計算用、1フレーム単位の情報を格納
 */
struct FrameData {
    int frameIndex;                              ///< フレーム番号
    std::vector<float> treatmentProbabilities;   ///< 推論スコア（15クラス分）
    std::vector<int> sceneBinaryLabels;          ///< シーンラベル（バイナリ、0～5主クラス）
    int sceneLabel = -1;                         ///< 平滑化されたラベル
    float sceneProb = 0.0f;                      ///< シーンラベルの確率
    float eventProbsSum = 0.0f;                  ///< イベントクラス確率の合計
};


/// =============================
/// @section スライディングウィンドウ管理
/// =============================

/**
 * @brief ウィンドウ内に画像・ラベルデータをバッファ管理
 */
class FrameWindow {
private:
	int windowSize;        ///< ウィンドウサイズ（フレーム数）
	int halfWindow;        ///< ウィンドウの中心フレームインデックス（windowSize / 2）

    std::deque<cv::Mat> windowedImages;         ///< ウィンドウ内の画像
    std::deque<FrameData> windowedFrameData;    ///< ウィンドウ内のフレームデータ

public:
    FrameWindow(int windowSize);

    /// 画像＋データをバッファに追加
    void push(const cv::Mat& image, const FrameData& data);

    /// ウィンドウ中心（解析対象）の画像・データ
    const cv::Mat& getCenterImage() const;
    const FrameData& getCenterData() const;

    /// ウィンドウ中心のシーンラベル書き換え
    void setCenterSceneLabel(int centerLabel);

    /// ウィンドウ内の全フレームのシーンラベル配列取得
    std::vector<std::vector<int>> getSceneLabels() const;

    /// バッファ状態確認
    const std::deque<FrameData>& getWindowedFrameData() const { return windowedFrameData; }

    // デバッグ出力
    void logFirstFrameData() const;
};


/// ===============================================
/// @section シーンラベル平滑化（スライディングウィンドウ）
/// ===============================================

/**
 * @brief 複数フレームのラベルを多数決で平滑化
 */
class SceneLabelSmoother {
private:
    int numSceneLabels;
    int windowSize;
    int halfWindow;
    int prevLabel = -1;

public:
    /// コンストラクタ：ラベル数とウィンドウサイズを指定
    SceneLabelSmoother(int numSceneLabels, int windowSize);

    /// ウィンドウ内ラベルから多数決で中心ラベルを返す
    std::optional<int> processSceneLabel(const std::vector<std::vector<int>>& windowSceneLabels);

    /// 中心フレームオフセット（windowSize / 2）
    int getWindowOffset() const;
};