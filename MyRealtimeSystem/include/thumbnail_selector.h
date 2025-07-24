#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <map>
#include <vector>
#include "sliding_window.h"


// ==================== サムネイル選定・管理クラス ====================
/// @brief サムネイル候補情報（スコア・フレーム情報付き）
struct ThumbnailCandidate {
    int frameIndex;                 ///< フレーム番号
    cv::Mat frame;                  ///< サムネイル画像
    float deepLearningScore;        ///< 深層学習によるスコア
    float highFrequencyScore;       ///< 高周波成分のスコア

    /// @brief サムネイルの総合スコア（2つの指標を掛け合わせる）
    float combinedScore() const {
        return deepLearningScore * highFrequencyScore;
    }

    /// @brief priority_queueで高スコア順に並べるための比較演算子
    bool operator<(const ThumbnailCandidate& o) const {
        return combinedScore() > o.combinedScore();  // 昇順にするため反転
    }
};


/// @brief 連続ラベル区間の情報＋Top-Kサムネイル候補
struct VideoSegment {
    int startFrameIndex = -1;                   ///< 区間開始フレーム
    int endFrameIndex = -1;                     ///< 区間終了フレーム
    int length = 0;                             ///< 区間の長さ（フレーム数）
    std::priority_queue<ThumbnailCandidate> topKThumbnails;  ///< Top-Kサムネイル候補
};


/// ==================== シーンセグメント管理クラス ====================
/// @brief 最長区間とTop-Kサムネイルをラベルごとに管理するクラス
class SceneSegmentManager {
private:
    int topK;                                  ///< サムネイル候補数
    int frameGap;                              ///< フレーム間隔の閾値
    int prevLabel = -1;                        ///< 前フレームのラベル

    std::map<int, VideoSegment> currentSegment; ///< 処理中の区間情報
    std::map<int, VideoSegment> longestSegment; ///< ラベルごとの最長区間情報

    /// @brief フレーム間隔を考慮してTop-K位サムネイルを抽出
    std::vector<ThumbnailCandidate> selectWithFrameGap(
        std::priority_queue<ThumbnailCandidate> candidates,
        int frameGap, int topK
    ) const;

    /// @brief 深層学習スコアを計算
    float computeDeepLearningScore(float sceneProb, float eventProbsSum);

	/// @brief 高周波エネルギーを計算
    float computeHighFrequencyEnergy(const cv::Mat& image);

public:
    /// @brief コンストラクタ
    SceneSegmentManager(int topK, int frameGap);

    /// @brief 1フレーム分の情報で区間を更新
    void update(const FrameData& data, const cv::Mat& image);

    /// @brief 最終区間を反映（updateの後に呼び出す）
    void finalize();

    /// @brief ラベルごとの最終Top-Kサムネイル一覧を返す
    std::map<int, std::vector<ThumbnailCandidate>> getFinalThumbnails() const;

    /// @brief 管理情報をログ出力する
    void logSummary() const;
};


// ==================== サムネイル可視化ユーティリティ ====================
/// @brief サムネイルをタイル状に合成してラベルごとに画像出力する
/// @param thumbsPerLabel   ラベルごとのサムネイル候補配列
/// @param savePath         保存先ファイルパス（"_label_0.png"等が付与される）
/// @param thumbWidth       サムネイル横幅[pixel]
/// @param thumbHeight      サムネイル縦幅[pixel]
/// @param gridCols         タイル合成時の列数
void visualizeThumbnailsPerLabel(
    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
    const std::string& savePath,
    int thumbWidth = 160, int thumbHeight = 120, int gridCols = 4
);

/// @brief ラベルごとの最終Top-Kサムネイル候補一覧をログ出力
void logFinalThumbnails(const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel);