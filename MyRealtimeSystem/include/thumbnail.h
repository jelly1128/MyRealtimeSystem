#pragma once
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>


//struct ThumbnailCandidate {
//    int frameIndex;             // フレーム番号
//    cv::Mat frame;              // フレーム画像
//    float deepLearningScore;    // 深層学習モデルによるスコア
//    float highFrequencyScore;   // 高周波エネルギーによるスコア
//
//    // 合成スコア計算式をメンバ関数で定義
//    float combinedScore() const {
//        //return highFrequencyScore;
//        //return deepLearningScore;
//        return deepLearningScore * highFrequencyScore;
//    }
//
//    // priority_queue用の比較演算子
//    bool operator<(const ThumbnailCandidate& o) const {
//        return combinedScore() > o.combinedScore();
//    }
//};
//
//
//struct VideoSegment {
//    int startFrameIndex = -1;                               // セグメントの開始フレームインデックス
//    int endFrameIndex = -1;                                 // セグメントの終了フレームインデックス
//    int length = 0;                                         // セグメントの長さ（フレーム数）
//    std::priority_queue<ThumbnailCandidate> topKThumbnails; // このセグメントの上位K個のサムネイル候補
//};


//float computeDeeplearningScore(float sceneProb, float );

float computeHighFrequencyEnergy(const cv::Mat& inputImg);

//std::vector<ThumbnailCandidate> selectThumbnailsWithFrameGap(
//    std::priority_queue<ThumbnailCandidate> topKThumbs,
//    int frameGap,
//    int topK
//);


// サムネイルをタイル状に合成して1枚の画像にする
//void visualizeThumbnailsPerLabel(
//    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
//    const std::string& savePath,
//    int thumbWidth = 160, int thumbHeight = 120, int gridCols = 4
//);