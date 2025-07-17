#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>


struct ThumbnailCandidate {
    int frameIndex;             // フレーム番号
    cv::Mat frame;              // フレーム画像
    float deepLearningScore;    // 深層学習モデルによるスコア
    float highFrequencyScore;   // 高周波エネルギーによるスコア

    // 合成スコア計算式をメンバ関数で定義
    float combinedScore() const {
        // 仮に平均で合成（実際はここを好きな式に変更可能！）
        // 例: return (deepLearningScore + highFrequencyScore) / 2.0;
        // 例: return deepLearningScore * 0.7 + highFrequencyScore * 0.3;
        // 実装時にここを調整！
        return (deepLearningScore + highFrequencyScore) / 2.0f;
    }

    // priority_queue用の比較演算子
    bool operator<(const ThumbnailCandidate& o) const {
        // priority_queueはデフォで「大きい順」にしたいのでこう書く
        return combinedScore() < o.combinedScore();
    }
};


struct VideoSegment {
    int startFrameIndex = -1;                               // セグメントの開始フレームインデックス
    int endFrameIndex = -1;                                 // セグメントの終了フレームインデックス
    int length = 0;                                  // セグメントの長さ（フレーム数）
    std::priority_queue<ThumbnailCandidate> topKThumbnails; // このセグメントの上位K個のサムネイル候補
};


float computeHighFrequencyEnergy(const cv::Mat& inputImg);

std::vector<ThumbnailCandidate> selectThumbnailsWithFrameGap(
    std::priority_queue<ThumbnailCandidate> topKThumbs,
    int frameGap,
    int topK
);