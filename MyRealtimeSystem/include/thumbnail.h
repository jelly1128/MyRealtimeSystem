#pragma once
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>


//struct ThumbnailCandidate {
//    int frameIndex;             // �t���[���ԍ�
//    cv::Mat frame;              // �t���[���摜
//    float deepLearningScore;    // �[�w�w�K���f���ɂ��X�R�A
//    float highFrequencyScore;   // �����g�G�l���M�[�ɂ��X�R�A
//
//    // �����X�R�A�v�Z���������o�֐��Œ�`
//    float combinedScore() const {
//        //return highFrequencyScore;
//        //return deepLearningScore;
//        return deepLearningScore * highFrequencyScore;
//    }
//
//    // priority_queue�p�̔�r���Z�q
//    bool operator<(const ThumbnailCandidate& o) const {
//        return combinedScore() > o.combinedScore();
//    }
//};
//
//
//struct VideoSegment {
//    int startFrameIndex = -1;                               // �Z�O�����g�̊J�n�t���[���C���f�b�N�X
//    int endFrameIndex = -1;                                 // �Z�O�����g�̏I���t���[���C���f�b�N�X
//    int length = 0;                                         // �Z�O�����g�̒����i�t���[�����j
//    std::priority_queue<ThumbnailCandidate> topKThumbnails; // ���̃Z�O�����g�̏��K�̃T���l�C�����
//};


//float computeDeeplearningScore(float sceneProb, float );

float computeHighFrequencyEnergy(const cv::Mat& inputImg);

//std::vector<ThumbnailCandidate> selectThumbnailsWithFrameGap(
//    std::priority_queue<ThumbnailCandidate> topKThumbs,
//    int frameGap,
//    int topK
//);


// �T���l�C�����^�C����ɍ�������1���̉摜�ɂ���
//void visualizeThumbnailsPerLabel(
//    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
//    const std::string& savePath,
//    int thumbWidth = 160, int thumbHeight = 120, int gridCols = 4
//);