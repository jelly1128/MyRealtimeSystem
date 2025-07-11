#include "video_loader.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
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


// �摜�O����
torch::Tensor preprocessFrame(
    const cv::Mat& frame,
    int inputWidth, int inputHeight, // ���T�C�Y��̃T�C�Y
    const cv::Rect& cropBox,         // �N���b�v�̈�i���w��Ȃ�S��)
    const cv::Mat& mask              // �}�X�N�摜�i��Ȃ疢�g�p�j
) {
    frame.clone();

    // --- (1) �}�X�N�K�p ---
    cv::Mat masked;
    if (!mask.empty() && mask.size() == frame.size()) {
        cv::bitwise_and(frame, frame, masked, mask);
    } else {
        masked = frame.clone();
    }

	// --- (2) �N���b�v ---
    cv::Mat cropped;
    if (cropBox.width > 0 && cropBox.height > 0 &&
        cropBox.x >= 0 && cropBox.y >= 0 &&
        cropBox.x + cropBox.width <= masked.cols &&
        cropBox.y + cropBox.height <= masked.rows) {
        cropped = masked(cropBox).clone();
	} else {
        cropped = masked.clone();  // �N���b�v���Ȃ��ꍇ�͑S��
    }

	// --- (3) ���T�C�Y ---
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));
	resized = resized.clone();  // ���T�C�Y��̉摜���m��

	// --- (4) �J���[�`�����l���ϊ� ---
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
	rgb = rgb.clone();  // �A�����̂��߂�clone

    // === �f�o�b�O�\�� ===
    //{
    //    cv::Mat debugImg, debugShow;
    //    rgb.convertTo(debugImg, CV_8U, 255.0); // float �� uchar
    //    cv::cvtColor(debugImg, debugShow, cv::COLOR_RGB2BGR); // RGB��BGR�ɖ߂�
    //    if (!debugShow.empty()) {
    //        cv::imshow("Debug RGB", debugShow);
    //        int key = cv::waitKey(0);
    //    }
    //    else {
    //        std::cerr << "debugShow is empty! Cannot show window." << std::endl;
    //    }
    //}
    // ==================

	// --- (5) Tensor�ϊ� ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC �� NCHW

    inputTensor = inputTensor.to(torch::kCUDA);

    return inputTensor;
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