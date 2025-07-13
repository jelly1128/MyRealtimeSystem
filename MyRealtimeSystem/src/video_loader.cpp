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
    // _����.png �܂��� ����.png �̂ǂ���ɂ��}�b�`
    std::regex pattern(R"((?:_|/)?(\d+)\.png$)");
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);  // �}�b�`�����ԍ��𐮐��ɕϊ�
    }
    return -1;  // ������Ȃ������ꍇ�i�擪�ɗ��Ȃ��悤�Ɂj
}


// �t�H���_����t���[����ǂݍ��ފ֐�
bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
    frames.clear();
    std::vector<cv::String> filepaths;

	//std::vector<int> frameNumbers; // for debug: �t���[���ԍ���ێ����邽�߂̃x�N�^�[

    cv::glob(folderPath + "/*.png", filepaths, false);
    if (filepaths.empty()) return false;

    // �t���[���ԍ��Ń\�[�g
    std::sort(filepaths.begin(), filepaths.end(), [](const cv::String& a, const cv::String& b) {
        return extractFrameNumber(a) < extractFrameNumber(b);
        });

    for (const auto& path : filepaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        int frameNum = extractFrameNumber(path);
        if (!img.empty()) {
            frames.push_back(img);
			//frameNumbers.push_back(frameNum); // for debug: �t���[���ԍ���ǉ�
        }
        else {
            std::cerr << "Failed to load image: " << path << std::endl;
        }
    }

	// for debug: �t���[���ԍ���\��
    /*std::cout << "Loaded frame numbers: ";
    for (auto num : frameNumbers) std::cout << num << " ";
    std::cout << std::endl;*/

    return !frames.empty();
}


// �摜�O����
torch::Tensor preprocessFrameForTreatment(
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
    /*cv::imshow("1. Masked", masked);
    std::cout << "Masked size: " << masked.cols << "x" << masked.rows << std::endl;

    cv::imshow("2. Crop", cropped);
    std::cout << "Cropped size: " << cropped.cols << "x" << cropped.rows << std::endl;

    cv::imshow("3. Resized", resized);
    std::cout << "Resized size: " << resized.cols << "x" << resized.rows << std::endl;

    cv::waitKey(0);*/
    // ==================

	// --- (5) Tensor�ϊ� ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC �� NCHW

    inputTensor = inputTensor.to(torch::kCUDA);

    return inputTensor;
}


// �摜�O����
torch::Tensor preprocessFrameForOrgan(
    const cv::Mat& frame,
    int inputWidth, int inputHeight, // ���T�C�Y��̃T�C�Y
	const int targetShort,           // �Z�ӂ̖ڕW�T�C�Y
    const cv::Rect& cropBox,         // �N���b�v�{�b�N�X�i�f�t�H���g�͑S�́j
    const cv::Mat& mask              // �}�X�N�摜�i��Ȃ疢�g�p�j
) {
    frame.clone();

    // --- (1) �}�X�N�K�p�iframe�S�̂�mask�Bmask�T�C�Y��frame�Ɠ����j ---
    cv::Mat masked;
    if (!mask.empty() && mask.size() == frame.size()) {
        cv::bitwise_and(frame, frame, masked, mask);
    }
    else {
        masked = frame.clone();
    }

    // --- (2) cropBox�ŃN���b�v�i�����ŕ������o�j ---
    cv::Mat croppedFirst;
    if (cropBox.width > 0 && cropBox.height > 0 &&
        cropBox.x >= 0 && cropBox.y >= 0 &&
        cropBox.x + cropBox.width <= masked.cols &&
        cropBox.y + cropBox.height <= masked.rows) {
        croppedFirst = masked(cropBox).clone();
    }
    else {
        croppedFirst = masked.clone();  // �N���b�v���Ȃ��ꍇ�͑S��
    }

    // --- (3) SmallestMaxSize�i�Z��=targetShort�ŃA�X�y�N�g��ێ����T�C�Y�j ---
    int origW = croppedFirst.cols;
    int origH = croppedFirst.rows;
    float scale = (float)targetShort / std::min(origW, origH);
    int resizedW = std::round(origW * scale);
    int resizedH = std::round(origH * scale);

    cv::Mat resized;
    cv::resize(croppedFirst, resized, cv::Size(resizedW, resizedH));

    // --- (4) ����inputWidth�~inputHeight�N���b�v ---
    int x = (resized.cols - inputWidth) / 2; // 338-224=114 �� x=57
    int y = (resized.rows - inputHeight) / 2; // 270-224=46 �� y=23

    cv::Rect centerCropBox(x, y, inputWidth, inputHeight);
    cv::Mat cropped = resized(centerCropBox).clone();

    // --- (5) �J���[�`�����l���ϊ� ---
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    rgb = rgb.clone();  // �A�����̂��߂�clone

    // === �f�o�b�O�\�� ===
    // �f�o�b�O�E�B���h�E�����ׂĕ\��
    /*cv::imshow("1. Masked", masked);
	std::cout << "Masked size: " << masked.cols << "x" << masked.rows << std::endl;

    cv::imshow("2. Cropped First", croppedFirst);
    std::cout << "Cropped size: " << croppedFirst.cols << "x" << croppedFirst.rows << std::endl;

    cv::imshow("3. Resized", resized);
    std::cout << "Resized size: " << resized.cols << "x" << resized.rows << std::endl;

    cv::imshow("4. Center Cropped", cropped);
    std::cout << "Cropped size after center crop: " << cropped.cols << "x" << cropped.rows << std::endl;

    cv::waitKey(0);*/
    // ==================

    // --- (6) Tensor�ϊ� ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC �� NCHW

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (7) ���K�� ---
    torch::Tensor mean = torch::tensor({ 0.5, 0.5, 0.5 }).view({ 1, 3, 1, 1 }).to(torch::kCUDA);
    torch::Tensor std = torch::tensor({ 0.5, 0.5, 0.5 }).view({ 1, 3, 1, 1 }).to(torch::kCUDA);
    inputTensor = (inputTensor - mean) / std;

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