#include "video_loader.h"
#include <opencv2/opencv.hpp>
#include <regex>

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


// フレーム番号を抽出する関数
int extractFrameNumber(const std::string& filename) {
    std::smatch match;
    std::regex pattern(R"(_(\d+)\.png$)");
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);  // マッチした番号を整数に変換
    }
    return -1;  // 見つからなかった場合（先頭に来ないように）
}

bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
    frames.clear();
    std::vector<cv::String> filepaths;

    cv::glob(folderPath + "/*.png", filepaths, false);
    if (filepaths.empty()) return false;

    // フレーム番号でソート
    std::sort(filepaths.begin(), filepaths.end(), [](const cv::String& a, const cv::String& b) {
        return extractFrameNumber(a) < extractFrameNumber(b);
        });

    for (const auto& path : filepaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            frames.push_back(img);
        }
        else {
            std::cerr << "Failed to load image: " << path << std::endl;
        }
    }

    return !frames.empty();
}


//bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
//    frames.clear();
//    std::vector<cv::String> filepaths;
//
//    // ワイルドカードで画像拡張子を指定（複数使うなら繰り返す）
//    cv::glob(folderPath + "/*.png", filepaths, false);  // PNG
//
//    if (filepaths.empty()) return false;
//
//    // ソート（ファイル名順）
//    std::sort(filepaths.begin(), filepaths.end());
//
//    for (const auto& path : filepaths) {
//        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
//        if (!img.empty()) {
//            frames.push_back(img);
//        }
//    }
//
//    return !frames.empty();
//}