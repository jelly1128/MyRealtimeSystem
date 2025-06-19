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
            frames.push_back(frame.clone());  // ���S�ɃR�s�[
        }
        frameCount++;
    }

    cap.release();
    return true;
}


// �t���[���ԍ��𒊏o����֐�
int extractFrameNumber(const std::string& filename) {
    std::smatch match;
    std::regex pattern(R"(_(\d+)\.png$)");
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);  // �}�b�`�����ԍ��𐮐��ɕϊ�
    }
    return -1;  // ������Ȃ������ꍇ�i�擪�ɗ��Ȃ��悤�Ɂj
}

bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
    frames.clear();
    std::vector<cv::String> filepaths;

    cv::glob(folderPath + "/*.png", filepaths, false);
    if (filepaths.empty()) return false;

    // �t���[���ԍ��Ń\�[�g
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
//    // ���C���h�J�[�h�ŉ摜�g���q���w��i�����g���Ȃ�J��Ԃ��j
//    cv::glob(folderPath + "/*.png", filepaths, false);  // PNG
//
//    if (filepaths.empty()) return false;
//
//    // �\�[�g�i�t�@�C�������j
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