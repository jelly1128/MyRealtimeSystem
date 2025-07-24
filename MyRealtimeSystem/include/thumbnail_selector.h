#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <map>
#include <vector>
#include "sliding_window.h"


// �T���l�C�������i�X�R�A�E�t���[�����t���j
struct ThumbnailCandidate {
    int frameIndex;
    cv::Mat frame;
    float deepLearningScore;
    float highFrequencyScore;

    float combinedScore() const {
        return deepLearningScore * highFrequencyScore;
    }

    // priority_queue �ō��X�R�A���ɕ��ׂ邽�߂̔�r���Z�q
    bool operator<(const ThumbnailCandidate& o) const {
        return combinedScore() > o.combinedScore();  // �����ɂ��邽�ߔ��]
    }
};

// �A�����x����Ԃ̏��{Top-K�T���l�C�����
struct VideoSegment {
    int startFrameIndex = -1;
    int endFrameIndex = -1;
    int length = 0;
    std::priority_queue<ThumbnailCandidate> topKThumbnails;
};

// �Œ���Ԃ�Top-K�T���l�C�������x�����ƂɊǗ�����N���X
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
    void finalize();  // �Ō�̃Z�O�����g�𔽉f
    std::map<int, std::vector<ThumbnailCandidate>> getFinalThumbnails() const;
    void logSummary() const;
};


// �T���l�C�����^�C����ɍ�������1���̉摜�ɂ���
void visualizeThumbnailsPerLabel(
    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
    const std::string& savePath,
    int thumbWidth = 160, int thumbHeight = 120, int gridCols = 4
);