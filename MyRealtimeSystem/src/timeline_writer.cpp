#include "timeline_writer.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm>
#include <string>


// �������imode:�ŕp�l�j�֐�
int getMode(const std::vector<int>& vec) {
    std::map<int, int> counts;
    for (auto v : vec) counts[v]++;
    int maxLabel = -1, maxCount = 0;
    for (auto&& [label, count] : counts) {
        if (count > maxCount || (count == maxCount && label < maxLabel)) {
            maxLabel = label;
            maxCount = count;
        }
    }
    return maxLabel;
}


bool drawTimelineImage(
    const std::vector<int>& labels,
    const std::string& savePath,
    int numSceneClasses,
	int timelineWidth,
	int timelineHeight
) {
    int numFrames = labels.size();
    if (numFrames == 0) return false;

    // �F��`�iBGR�j
    std::array<cv::Scalar, 7> classColors = {
        cv::Scalar(195, 195, 254),  // ���F��
        cv::Scalar(38, 66, 204),    // ���S�[��
        cv::Scalar(177, 103, 57),   // �C���f�B�S�J���~��
        cv::Scalar(53, 165, 96),    // ���ш��
        cv::Scalar(72, 65, 86),     // ����
        cv::Scalar(183, 190, 159),  // �o�P�c
        cv::Scalar(148, 148, 148),  // unknown
    };

    // �摜�쐬�i���w�i�j
    cv::Mat image(timelineHeight, timelineWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // �_�E���T���v�����O
    std::vector<int> sampledLabels(timelineWidth, 6); // 6=unknown
    double samplingRatio = numFrames / static_cast<double>(timelineWidth);

    // --- �^�C�����C���`�� ---
    for (int x = 0; x < timelineWidth; ++x) {
        int idx_start = static_cast<int>(x * samplingRatio);
        int idx_end = static_cast<int>((x + 1) * samplingRatio);
        if (idx_end > numFrames) idx_end = numFrames;
        if (idx_start >= idx_end) {
            sampledLabels[x] = (x > 0) ? sampledLabels[x - 1] : 6;
            continue;
        }
        std::vector<int> section(labels.begin() + idx_start, labels.begin() + idx_end);
        int mode = getMode(section);
        if (mode < 0 || mode >= numSceneClasses) mode = 6;
        sampledLabels[x] = mode;
    }

    // �^�C�����C���{�̕`��
    for (int x = 0; x < timelineWidth; ++x) {
        cv::rectangle(
            image,
            cv::Rect(x, 0, 1, timelineHeight),
            classColors[sampledLabels[x]],
            cv::FILLED
        );
    }

    return cv::imwrite(savePath, image);
}
