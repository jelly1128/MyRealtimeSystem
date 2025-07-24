#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <map>
#include <vector>
#include "sliding_window.h"


// サムネイル候補情報（スコア・フレーム情報付き）
struct ThumbnailCandidate {
    int frameIndex;
    cv::Mat frame;
    float deepLearningScore;
    float highFrequencyScore;

    float combinedScore() const {
        return deepLearningScore * highFrequencyScore;
    }

    // priority_queue で高スコア順に並べるための比較演算子
    bool operator<(const ThumbnailCandidate& o) const {
        return combinedScore() > o.combinedScore();  // 昇順にするため反転
    }
};

// 連続ラベル区間の情報＋Top-Kサムネイル候補
struct VideoSegment {
    int startFrameIndex = -1;
    int endFrameIndex = -1;
    int length = 0;
    std::priority_queue<ThumbnailCandidate> topKThumbnails;
};

// 最長区間とTop-Kサムネイルをラベルごとに管理するクラス
class SceneSegmentManager {
private:
    int topK;
    int frameGap;
    int prevLabel = -1;

    std::map<int, VideoSegment> currentSegment;
    std::map<int, VideoSegment> longestSegment;

    std::vector<ThumbnailCandidate> selectWithFrameGap(
        std::priority_queue<ThumbnailCandidate> candidates,
        int frameGap, int topK
    ) const;

    float computeDeeplearningScore(float sceneProb, float eventProbsSum);
    float computeHighFrequencyEnergy(const cv::Mat& image);

public:
    SceneSegmentManager(int topK, int frameGap);
    void update(const FrameData& data, const cv::Mat& image);
    void finalize();  // 最後のセグメントを反映
    std::map<int, std::vector<ThumbnailCandidate>> getFinalThumbnails() const;
    void logSummary() const;
};


// サムネイルをタイル状に合成して1枚の画像にする
void visualizeThumbnailsPerLabel(
    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
    const std::string& savePath,
    int thumbWidth = 160, int thumbHeight = 120, int gridCols = 4
);