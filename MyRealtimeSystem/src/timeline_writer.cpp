#include "timeline_writer.h"
#include <opencv2/opencv.hpp>
#include <array>

bool drawTimelineImage(
    const std::vector<int>& labels,
    const std::string& savePath,
    int numClasses,
    int frameWidth,
    int rowHeight
) {
    int numFrames = labels.size();
    int width = numFrames * frameWidth;
    int height = rowHeight;

    if (numFrames == 0) return false;

    // 色定義（BGR）
    std::array<cv::Scalar, 6> classColors = {
        cv::Scalar(195, 195, 254),  // white
        cv::Scalar(38, 66, 204),    // lugol
        cv::Scalar(177, 103, 57),   // indigo
        cv::Scalar(53, 165, 96),    // nbi
        cv::Scalar(72, 65, 86),     // outside
        cv::Scalar(183, 190, 159)   // bucket
    };

    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));  // 白背景

    for (int i = 0; i < numFrames; ++i) {
        int x = i * frameWidth;
        int label = labels[i];
        if (label < 0 || label >= numClasses) continue;

        cv::rectangle(
            image,
            cv::Rect(x, 0, frameWidth, rowHeight),
            classColors[label],
            cv::FILLED
        );
    }

    return cv::imwrite(savePath, image);
}
