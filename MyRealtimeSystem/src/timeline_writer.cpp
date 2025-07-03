#include "timeline_writer.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm>
#include <string>

bool drawTimelineImage(
    const std::vector<int>& labels,
    const std::string& savePath,
    int numClasses
) {
    int numFrames = labels.size();
    if (numFrames == 0) return false;

    // --- Auto-adjust timeline dimensions ---
    int timeline_width = numFrames;
    int timeline_height = std::max(10, numFrames / 10);   // タイムライン本体
    int tick_height = std::max(5, timeline_height / 10);  // メモリ線の長さ
    int label_height = 12;                                // ラベル表示の高さ

    int image_height = timeline_height + tick_height + label_height;

    // 色定義（BGR）
    std::array<cv::Scalar, 7> classColors = {
        cv::Scalar(195, 195, 254),  // white
        cv::Scalar(38, 66, 204),    // lugol
        cv::Scalar(177, 103, 57),   // indigo
        cv::Scalar(53, 165, 96),    // nbi
        cv::Scalar(72, 65, 86),     // outside
        cv::Scalar(183, 190, 159),  // bucket
        cv::Scalar(148, 148, 148),  // unknown
    };

    // 画像作成（白背景）
    cv::Mat image(image_height, timeline_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // --- タイムライン描画（上部） ---
    for (int i = 0; i < numFrames; ++i) {
        int x1 = i * timeline_width / numFrames;
        int x2 = (i + 1) * timeline_width / numFrames;
        int label = labels[i];
        if (label < 0 || label >= numClasses) label = 6; // unknown

        cv::rectangle(
            image,
            cv::Rect(x1, 0, x2 - x1, timeline_height),
            classColors[label],
            cv::FILLED
        );
    }

    // --- メモリ線とラベル描画 ---
    int fps = 3;
    int tickIntervalSec = 60;                  // 60秒ごと
    int tickInterval = fps * tickIntervalSec;  // = 180フレーム

    cv::Scalar tickColor(0, 0, 0);             // 黒
    int tickThickness = 2;

    int y1 = timeline_height;
    int y2 = timeline_height + tick_height;
    int textY = y2 + label_height - 2;

    for (int i = tickInterval; i < numFrames; i += tickInterval) {
        int x = i * timeline_width / numFrames;

        // メモリ線
        cv::line(image, cv::Point(x, y1), cv::Point(x, y2), tickColor, tickThickness);

        // フレーム数ラベル (要検討)
        std::string label = std::to_string(i);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
        int textX = x - textSize.width / 2;

        cv::putText(
            image,
            label,
            cv::Point(textX, textY),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            tickColor,
            2
        );
    }

    return cv::imwrite(savePath, image);
}
