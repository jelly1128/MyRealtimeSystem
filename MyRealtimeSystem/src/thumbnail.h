#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>


struct ThumbnailCandidate {
    int frameIndex;             // �t���[���ԍ�
    cv::Mat frame;              // �t���[���摜
    float deepLearningScore;    // �[�w�w�K���f���ɂ��X�R�A
    float highFrequencyScore;   // �����g�G�l���M�[�ɂ��X�R�A

    // �����X�R�A�v�Z���������o�֐��Œ�`
    float combinedScore() const {
        // ���ɕ��ςō����i���ۂ͂������D���Ȏ��ɕύX�\�I�j
        // ��: return (deepLearningScore + highFrequencyScore) / 2.0;
        //return std::min(deepLearningScore, highFrequencyScore);
        //return highFrequencyScore;
        //return deepLearningScore;
        return deepLearningScore * highFrequencyScore;
    }

    // priority_queue�p�̔�r���Z�q
    bool operator<(const ThumbnailCandidate& o) const {
        return combinedScore() > o.combinedScore();
    }
};


struct VideoSegment {
    int startFrameIndex = -1;                               // �Z�O�����g�̊J�n�t���[���C���f�b�N�X
    int endFrameIndex = -1;                                 // �Z�O�����g�̏I���t���[���C���f�b�N�X
    int length = 0;                                  // �Z�O�����g�̒����i�t���[�����j
    std::priority_queue<ThumbnailCandidate> topKThumbnails; // ���̃Z�O�����g�̏��K�̃T���l�C�����
};


float computeHighFrequencyEnergy(const cv::Mat& inputImg);

std::vector<ThumbnailCandidate> selectThumbnailsWithFrameGap(
    std::priority_queue<ThumbnailCandidate> topKThumbs,
    int frameGap,
    int topK
);