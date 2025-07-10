#include "video_loader.h"
#include <opencv2/opencv.hpp>
#include <regex>


// ���悩��t���[����ǂݍ��ފ֐�
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


// �t�H���_����t���[����ǂݍ��ފ֐�
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


// �ǂݍ��񂾉摜��\������(�f�o�b�N�p)
void showFrames(const std::vector<cv::Mat>& frames) {
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::imshow("Frame", frames[i]);
        int key = cv::waitKey(500); // 500ms���ƂɎ��̉摜��
        if (key == 27) break; // ESC�L�[�Œ��f
    }
    cv::destroyAllWindows();
}