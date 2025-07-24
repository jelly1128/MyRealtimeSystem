#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <map>
#include <vector>
#include "sliding_window.h"


// ==================== �T���l�C���I��E�Ǘ��N���X ====================
/// @brief �T���l�C�������i�X�R�A�E�t���[�����t���j
struct ThumbnailCandidate {
    int frameIndex;                 ///< �t���[���ԍ�
    cv::Mat frame;                  ///< �T���l�C���摜
    float deepLearningScore;        ///< �[�w�w�K�ɂ��X�R�A
    float highFrequencyScore;       ///< �����g�����̃X�R�A

    /// @brief �T���l�C���̑����X�R�A�i2�̎w�W���|�����킹��j
    float combinedScore() const {
        return deepLearningScore * highFrequencyScore;
    }

    /// @brief priority_queue�ō��X�R�A���ɕ��ׂ邽�߂̔�r���Z�q
    bool operator<(const ThumbnailCandidate& o) const {
        return combinedScore() > o.combinedScore();  // �����ɂ��邽�ߔ��]
    }
};


/// @brief �A�����x����Ԃ̏��{Top-K�T���l�C�����
struct VideoSegment {
    int startFrameIndex = -1;                   ///< ��ԊJ�n�t���[��
    int endFrameIndex = -1;                     ///< ��ԏI���t���[��
    int length = 0;                             ///< ��Ԃ̒����i�t���[�����j
    std::priority_queue<ThumbnailCandidate> topKThumbnails;  ///< Top-K�T���l�C�����
};


/// ==================== �V�[���Z�O�����g�Ǘ��N���X ====================
/// @brief �Œ���Ԃ�Top-K�T���l�C�������x�����ƂɊǗ�����N���X
class SceneSegmentManager {
private:
    int topK;                                  ///< �T���l�C����␔
    int frameGap;                              ///< �t���[���Ԋu��臒l
    int prevLabel = -1;                        ///< �O�t���[���̃��x��

    std::map<int, VideoSegment> currentSegment; ///< �������̋�ԏ��
    std::map<int, VideoSegment> longestSegment; ///< ���x�����Ƃ̍Œ���ԏ��

    /// @brief �t���[���Ԋu���l������Top-K�ʃT���l�C���𒊏o
    std::vector<ThumbnailCandidate> selectWithFrameGap(
        std::priority_queue<ThumbnailCandidate> candidates,
        int frameGap, int topK
    ) const;

    /// @brief �[�w�w�K�X�R�A���v�Z
    float computeDeepLearningScore(float sceneProb, float eventProbsSum);

	/// @brief �����g�G�l���M�[���v�Z
    float computeHighFrequencyEnergy(const cv::Mat& image);

public:
    /// @brief �R���X�g���N�^
    SceneSegmentManager(int topK, int frameGap);

    /// @brief 1�t���[�����̏��ŋ�Ԃ��X�V
    void update(const FrameData& data, const cv::Mat& image);

    /// @brief �ŏI��Ԃ𔽉f�iupdate�̌�ɌĂяo���j
    void finalize();

    /// @brief ���x�����Ƃ̍ŏITop-K�T���l�C���ꗗ��Ԃ�
    std::map<int, std::vector<ThumbnailCandidate>> getFinalThumbnails() const;

    /// @brief �Ǘ��������O�o�͂���
    void logSummary() const;
};


// ==================== �T���l�C���������[�e�B���e�B ====================
/// @brief �T���l�C�����^�C����ɍ������ă��x�����Ƃɉ摜�o�͂���
/// @param thumbsPerLabel   ���x�����Ƃ̃T���l�C�����z��
/// @param savePath         �ۑ���t�@�C���p�X�i"_label_0.png"�����t�^�����j
/// @param thumbWidth       �T���l�C������[pixel]
/// @param thumbHeight      �T���l�C���c��[pixel]
/// @param gridCols         �^�C���������̗�
void visualizeThumbnailsPerLabel(
    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
    const std::string& savePath,
    int thumbWidth = 160, int thumbHeight = 120, int gridCols = 4
);

/// @brief ���x�����Ƃ̍ŏITop-K�T���l�C�����ꗗ�����O�o��
void logFinalThumbnails(const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel);