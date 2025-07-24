#include "thumbnail_selector.h"
#include "debug.h"
#include <algorithm>
#include <set>


// ==================== SceneSegmentManager ���� ====================

/**
 * @brief �R���X�g���N�^
 * @param topK �T���l�C����␔
 * @param frameGap �T���l�C���̍ŏ��t���[���Ԋu
 */
SceneSegmentManager::SceneSegmentManager(int topK, int frameGap)
    :topK(topK), frameGap(frameGap) {
}


/**
 * @brief �[�w�w�K�X�R�A���v�Z
 * @param sceneProb �V�[���̊m��
 * @param eventProbsSum �C�x���g�N���X�̊m���̍��v
 * @return �[�w�w�K�X�R�A
 */
float SceneSegmentManager::computeDeepLearningScore(float sceneProb, float eventProbsSum) {
    // �[�w�w�K�X�R�A�̌v�Z��
    return sceneProb - eventProbsSum;
}


/**
 * @brief �����g�G�l���M�[���v�Z
 * @param image ���͉摜
 * @return �����g�G�l���M�[�̔䗦
 */
float SceneSegmentManager::computeHighFrequencyEnergy(const cv::Mat& image) {
    // 1. �O���[�X�P�[���ϊ�
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    }
    else {
        gray = image.clone();
    }
    gray.convertTo(gray, CV_32F);

    // 2. DFT�i���f���o�́j
    cv::Mat dftImg;
    cv::dft(gray, dftImg, cv::DFT_COMPLEX_OUTPUT);

    // 3. FFTShift�i����g�����𒆉��Ɂj
    int cx = dftImg.cols / 2;
    int cy = dftImg.rows / 2;
    cv::Mat q0(dftImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(dftImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(dftImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(dftImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // 4. �����g�}�X�N�쐬�i����g���S��0�Ɂj
    int radius = std::min(gray.rows, gray.cols) / 8;
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
    cv::circle(mask, cv::Point(cx, cy), radius, cv::Scalar(0), -1);

    // 4. �����g�}�X�N�쐬�i���S���l�p�`�ŃJ�b�g�j
    //int radius = std::min(gray.rows, gray.cols) / 8;
    //cv::Mat mask = cv::Mat::ones(gray.rows, gray.cols, CV_32F);
    //cv::Rect lowFreqBox(cx - radius, cy - radius, radius * 2, radius * 2);
    //mask(lowFreqBox) = 0.0f;  // ���S�̒���g�̈��0�ɂ���i�l�p�`�j

    // 5. �G�l���M�[�v�Z
    std::vector<cv::Mat> channels(2);
    cv::split(dftImg, channels);
    cv::Mat mag;
    cv::magnitude(channels[0], channels[1], mag);
    cv::pow(mag, 2, mag);  // �G�l���M�[ = �U��^2

    double totalEnergy = cv::sum(mag)[0];

    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F);
    cv::Mat highFreqOnly;
    cv::multiply(mag, maskFloat, highFreqOnly);
    double highFreqEnergy = cv::sum(highFreqOnly)[0];

    // 6. ���K�����ĕԂ�
    return totalEnergy > 0 ? static_cast<float>(highFreqEnergy / totalEnergy) : 0.0f;
}


/**
 * @brief 1�t���[�����i�X���C�f�B���O�E�B���h�E�̒��S�t���[���j�̃T���l�C�������Ǘ��E�X�V
 * @param data �t���[���f�[�^
 * @param image ���͉摜
 */
void SceneSegmentManager::update(const FrameData& data, const cv::Mat& image) {
	// �P�t���[�����̃f�[�^���T���l�C�����Ƃ��ēo�^
    int label = data.sceneLabel;
	float deepLearningScore = computeDeepLearningScore(data.sceneProb, data.eventProbsSum);
    float highFrequencyScore = computeHighFrequencyEnergy(image);

    ThumbnailCandidate candidate{ data.frameIndex, image.clone(), deepLearningScore, highFrequencyScore };

    // ���x�����ς�����ꍇ�A�O�̋�Ԃ��m��
    if (prevLabel != -1 && label != prevLabel) {
        auto& prevSeg = currentSegment[prevLabel];
        if (prevSeg.length > longestSegment[prevLabel].length) {
            longestSegment[prevLabel] = prevSeg;
        }
        currentSegment.erase(prevLabel);
    }

    // �Z�O�����g���X�V
    auto& seg = currentSegment[label];
    if (seg.length == 0) {
        seg.startFrameIndex = data.frameIndex;
    }
    seg.endFrameIndex = data.frameIndex;
    seg.length++;
    seg.topKThumbnails.push(candidate);
    if (seg.topKThumbnails.size() > topK) seg.topKThumbnails.pop();

    prevLabel = label;
}


/**
 * @brief �Œ���Ԃ��m��
 */
void SceneSegmentManager::finalize() {
    for (const auto& [label, seg] : currentSegment) {
        if (seg.length > longestSegment[label].length) {
            longestSegment[label] = seg;
        }
    }
    currentSegment.clear();
}


/**
 * @brief ���x�����Ƃ̍ŏITop-K�T���l�C�������擾
 */
std::map<int, std::vector<ThumbnailCandidate>> SceneSegmentManager::getFinalThumbnails() const {
    std::map<int, std::vector<ThumbnailCandidate>> result;
    for (const auto& [label, seg] : longestSegment) {
        result[label] = selectWithFrameGap(seg.topKThumbnails, frameGap, topK);
    }
    return result;
}

/**
 * @brief �t���[���Ԋu���l������Top-K����I��
 */
std::vector<ThumbnailCandidate> SceneSegmentManager::selectWithFrameGap(
    std::priority_queue<ThumbnailCandidate> candidates,
    int frameGap, int topK) const
{
    // priority_queue �� vector�i���X�R�A���j
    std::vector<ThumbnailCandidate> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.combinedScore() > b.combinedScore();
    });

	// �t���[���Ԋu���l�����đI��
    std::vector<ThumbnailCandidate> selected;
    std::set<int> usedFrames;

    for (const auto& cand : sorted) {
        bool tooClose = false;
        for (int idx : usedFrames) {
            if (std::abs(cand.frameIndex - idx) < frameGap) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            selected.push_back(cand);
            usedFrames.insert(cand.frameIndex);
            if (selected.size() == topK) break;
        }
    }
    return selected;
}


/**
 * @brief ���x�����Ƃ̍Œ���ԂƃT���l�C���������O�o��
 */
void SceneSegmentManager::logSummary() const {
    log("=== ���x�����Ƃ̍Œ���� ===", true);
    for (const auto& [label, seg] : longestSegment) {
        log("Label " + std::to_string(label)
            + " [" + std::to_string(seg.startFrameIndex) + "," + std::to_string(seg.endFrameIndex) + "]"
            + " (length=" + std::to_string(seg.length) + ")", true);

        // �T���l�C�����̃X�R�A���~���ŏo��
        std::priority_queue<ThumbnailCandidate> thumbs = seg.topKThumbnails;
        std::vector<ThumbnailCandidate> thumbsVec;
        while (!thumbs.empty()) {
            thumbsVec.push_back(thumbs.top());
            thumbs.pop();
        }
        std::sort(thumbsVec.begin(), thumbsVec.end(), [](const auto& a, const auto& b) {
            return a.combinedScore() > b.combinedScore();
            });
        for (const auto& cand : thumbsVec) {
            log("  Frame " + std::to_string(cand.frameIndex)
                + ", DLScore=" + std::to_string(cand.deepLearningScore)
                + ", HiFreq=" + std::to_string(cand.highFrequencyScore)
                + ", Score=" + std::to_string(cand.combinedScore()), true);
        }
    }
}

// ==================== �T���l�C���������[�e�B���e�B ====================

/**
 * @brief �T���l�C���摜���O���b�h��ɍ���
 * @param thumbs      �T���l�C���摜�z��
 * @param thumbWidth  �T���l�C����[pixel]
 * @param thumbHeight �T���l�C������[pixel]
 * @param gridCols    �O���b�h�񐔁i0�Ŏ����v�Z�j
 * @param gridRows    �O���b�h�s���i0�Ŏ����v�Z�j
 * @param margin      �T���l�C���ԃ}�[�W��[pixel]
 * @param bgColor     �w�i�F
 * @return �����摜
 */
cv::Mat createThumbnailTile(
    const std::vector<cv::Mat>& thumbs,
    int thumbWidth = 160, int thumbHeight = 120,
    int gridCols = 0, int gridRows = 0,
    int margin = 8, cv::Scalar bgColor = cv::Scalar(240, 240, 240)
) {
    int numThumbs = thumbs.size();
    if (gridCols == 0) gridCols = std::ceil(std::sqrt(numThumbs));
    if (gridRows == 0) gridRows = std::ceil((float)numThumbs / gridCols);

    int outWidth = thumbWidth * gridCols + margin * (gridCols + 1);
    int outHeight = thumbHeight * gridRows + margin * (gridRows + 1);
    cv::Mat canvas(outHeight, outWidth, CV_8UC3, bgColor);

    for (int i = 0; i < numThumbs; ++i) {
        int row = i / gridCols;
        int col = i % gridCols;
        int x = margin + col * (thumbWidth + margin);
        int y = margin + row * (thumbHeight + margin);

        if (!thumbs[i].empty()) {
            cv::Mat thumbResized;
            cv::resize(thumbs[i], thumbResized, cv::Size(thumbWidth, thumbHeight));
            thumbResized.copyTo(canvas(cv::Rect(x, y, thumbWidth, thumbHeight)));
        }
    }
    return canvas;
}


/**
 * @brief ���x�����Ƃ̃T���l�C�����摜�ɍ������ۑ�����
 * @param thumbsPerLabel ���x�����Ƃ̃T���l�C���z��
 * @param savePath       �ۑ���p�X�i"_label_0.png"�����t�^�����j
 * @param thumbWidth     �T���l�C����[pixel]
 * @param thumbHeight    �T���l�C������[pixel]
 * @param gridCols       �O���b�h��
 */
void visualizeThumbnailsPerLabel(
    const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel,
	const std::string& savePath,
    int thumbWidth, int thumbHeight, int gridCols
) {
    for (const auto& [label, thumbs] : thumbsPerLabel) {
        std::vector<cv::Mat> thumbImgs;
        for (const auto& cand : thumbs) {
            if (!cand.frame.empty()) {
                cv::Mat imgBGR;
                if (cand.frame.type() == CV_32FC3) {
                    cand.frame.convertTo(imgBGR, CV_8UC3, 255.0); // 0�`1��0�`255
                }
                else {
                    imgBGR = cand.frame.clone();
                }
                // �K�v�Ȃ�BGR�ϊ�
                cv::cvtColor(imgBGR, imgBGR, cv::COLOR_RGB2BGR);
                thumbImgs.push_back(imgBGR);
            }
        }
        if (thumbImgs.empty()) continue;

        cv::Mat tile = createThumbnailTile(thumbImgs, thumbWidth, thumbHeight, gridCols);

        // �摜�\��
        /*std::string winName = "Label " + std::to_string(label) + " Thumbnails";
        cv::imshow(winName, tile);
        cv::waitKey(0);*/

        // �摜�ۑ�
        cv::imwrite(savePath + "_label_" + std::to_string(label) + ".png", tile);
    }
}


void logFinalThumbnails(const std::map<int, std::vector<ThumbnailCandidate>>& thumbsPerLabel)
{
    log("===== ���x�����Ƃ̍ŏITop-K�T���l�C�����ꗗ =====", true);
    for (const auto& [label, thumbs] : thumbsPerLabel) {
        log("���x��: " + std::to_string(label) + "�i����: " + std::to_string(thumbs.size()) + "�j", true);
        int i = 0;
        for (const auto& cand : thumbs) {
            log("  [" + std::to_string(i++) +
                "] �t���[���ԍ�: " + std::to_string(cand.frameIndex) +
                ", �[�w�w�K�X�R�A: " + std::to_string(cand.deepLearningScore) +
                ", �����g�X�R�A: " + std::to_string(cand.highFrequencyScore) +
                ", ���Z�X�R�A: " + std::to_string(cand.combinedScore()), true);
        }
    }
    log("===============================================", true);
}