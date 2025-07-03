#include "timeline_writer.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm> // for std::max

bool drawTimelineImage(
    const std::vector<int>& labels,
    const std::string& savePath,
    int numClasses
) {
    int numFrames = labels.size();
    if (numFrames == 0) return false;

    // --- Auto-adjust timeline dimensions ---
    int timeline_width = numFrames;
    int timeline_height = std::max(10, numFrames / 10); // Avoid height=0

    // �F��`�iBGR�j
    std::array<cv::Scalar, 7> classColors = {
        cv::Scalar(195, 195, 254),  // white
        cv::Scalar(38, 66, 204),    // lugol
        cv::Scalar(177, 103, 57),   // indigo
        cv::Scalar(53, 165, 96),    // nbi
        cv::Scalar(72, 65, 86),     // outside
        cv::Scalar(183, 190, 159),  // bucket
		cv::Scalar(148, 148, 148),  // unknown
		// �ǉ��̃N���X���K�v�ȏꍇ�͂����ɒǉ�
    };

    cv::Mat image(timeline_height, timeline_width, CV_8UC3, cv::Scalar(255, 255, 255));  // ���w�i

    // �e�t���[���̕��������v�Z
    for (int i = 0; i < numFrames; ++i) {
        int x1 = i * timeline_width / numFrames;
        int x2 = (i + 1) * timeline_width / numFrames;
        int label = labels[i];
        if (label < 0 || label >= numClasses) {
            label = 6; // �f�t�H���g�̃N���X�i��: "unknown"�j���g�p
        }

        cv::rectangle(
            image,
            cv::Rect(x1, 0, x2 - x1, timeline_height),
            classColors[label],
            cv::FILLED
        );
    }

    return cv::imwrite(savePath, image);
}
