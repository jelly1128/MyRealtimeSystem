#include "video_loader.h"
#include <opencv2/opencv.hpp>

bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        return false;
    }

    int frameCount = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frameCount % frameInterval == 0) {
            frames.push_back(frame.clone());  // 安全にコピー
        }
        frameCount++;
    }

    cap.release();
    return true;
}


bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
    frames.clear();
    std::vector<cv::String> filepaths;

    // ワイルドカードで画像拡張子を指定（複数使うなら繰り返す）
    cv::glob(folderPath + "/*.png", filepaths, false);  // PNG

    if (filepaths.empty()) return false;

    // ソート（ファイル名順）
    std::sort(filepaths.begin(), filepaths.end());

    for (const auto& path : filepaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            frames.push_back(img);
        }
    }

    return !frames.empty();
}