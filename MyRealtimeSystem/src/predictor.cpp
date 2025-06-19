#include "predictor.h"
#include <torch/torch.h>
#include <iomanip>  // std::setprecision, std::fixed, std::scientific�p

bool loadModel(const std::string& modelPath, torch::jit::script::Module& model) {
    try {
        model = torch::jit::load(modelPath);
        model.to(torch::kCUDA);
        model.eval();
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "���f���̓ǂݍ��݂Ɏ��s���܂���: " << e.what() << std::endl;
        return false;
    }
}

static bool saved = false;  // �f�o�b�O�p�t���O

std::vector<float> predictFrame(const cv::Mat& frame, torch::jit::script::Module& model, int inputWidth, int inputHeight) {
    // --- (0) �}�X�N�K�p�i�O�����j ---
    //cv::Mat masked;
    //cv::Mat mask = cv::imread("images/mask.png", cv::IMREAD_GRAYSCALE);
    //if (mask.size() != frame.size()) {
    //    std::cerr << "�}�X�N�T�C�Y�����͉摜�ƈ�v���܂���" << std::endl;
    //    // return ... or resize(mask, ...)
    //}
    //cv::bitwise_and(frame, frame, masked, mask);

    // --- (1) �N���b�v ---
    //cv::Rect cropBox(330, 25, 1260, 970);
    //cv::Mat cropped = masked(cropBox).clone();

    //// --- (2) ���T�C�Y + �J���[�`�����l���ϊ� ---
    //cv::Mat resized, rgb;
    //cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));  // e.g. 224x224

    // ���T�C�Y�܂łł��Ă���ꍇ
	cv::Mat resized, rgb;
    resized = frame.clone();

    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    if (!rgb.isContinuous()) {
        rgb = rgb.clone();  // from_blob�̑O�ɘA����
    }

    // --- (3) Tensor�ϊ� ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC �� NCHW

    // --- (4) �W�����iImageNet�p�j---
    //inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
    //inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
    //inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

	// �f�o�b�O�p�ɉ摜��ۑ�
    if (!saved) {
		cv::imwrite("debug_frame.png", frame);
		//cv::imwrite("debug_masked.png", masked);
		cv::imwrite("debug_rgb.png", rgb);
        //cv::imwrite("debug_cropped.png", cropped);
        cv::imwrite("debug_resized.png", resized);
        saved = true;
    }

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (5) ���_ ---
    torch::NoGradGuard no_grad;
    auto output = model.forward({ inputTensor }).toTensor();
    auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU);

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}


//std::vector<float> predictFrame(const cv::Mat& frame, torch::jit::script::Module& model, int inputWidth, int inputHeight) {
//    static bool saved = false;
//
//    // --- (1) PIL��convert("RGB")�ɑ������鏈�� ---
//    cv::Mat rgb;
//    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);  // OpenCV��BGR�APIL��RGB
//
//    // --- (2) ���T�C�Y�i�K�v�ɉ����āj ---
//    cv::Mat resized;
//    if (rgb.size() != cv::Size(inputWidth, inputHeight)) {
//        cv::resize(rgb, resized, cv::Size(inputWidth, inputHeight));
//    }
//    else {
//        resized = rgb.clone();
//    }
//
//    // --- (3) ToTensor()�ɑ������鏈�� ---
//    // PIL��ToTensor()��: (H, W, C) -> (C, H, W) ���� [0,255] -> [0,1]
//    resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // [0,255] -> [0,1]
//
//    if (!resized.isContinuous()) {
//        resized = resized.clone();
//    }
//
//    // --- (4) Tensor�ϊ�: HWC -> NCHW ---
//    torch::Tensor inputTensor = torch::from_blob(
//        resized.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
//    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC -> NCHW
//
//    // --- (5) ImageNet�W�����͍s��Ȃ��I�iPython��������Ă��Ȃ����߁j ---
//    // inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);  // �R�����g�A�E�g
//    // inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);  // �R�����g�A�E�g
//    // inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);  // �R�����g�A�E�g
//
//    // --- (6) �f�o�b�O�p�ۑ��E���v�o�� ---
//    if (!saved) {
//        std::cout << "=== C++ Debug Info ===" << std::endl;
//        std::cout << "Input frame size: " << frame.size() << ", channels: " << frame.channels() << std::endl;
//        std::cout << "Target size: " << inputWidth << "x" << inputHeight << std::endl;
//
//        // �e�i�K�ł̉摜���
//        std::cout << "Original frame - dtype: " << frame.type() << ", range: [" << frame.at<cv::Vec3b>(0, 0) << "]" << std::endl;
//        std::cout << "RGB frame - dtype: " << rgb.type() << std::endl;
//        std::cout << "Resized frame - dtype: " << resized.type() << ", size: " << resized.size() << std::endl;
//
//        // OpenCV Mat �̓��v���
//        cv::Scalar mean_val, stddev_val;
//        cv::meanStdDev(resized, mean_val, stddev_val);
//        std::cout << "OpenCV Mat stats - Mean: " << mean_val << ", StdDev: " << stddev_val << std::endl;
//
//        // Tensor���
//        std::cout << "Tensor shape: " << inputTensor.sizes() << std::endl;
//        std::cout << "Tensor dtype: " << inputTensor.dtype() << std::endl;
//
//        // Tensor�͈̔͊m�F
//        auto tensor_min = inputTensor.min().item<float>();
//        auto tensor_max = inputTensor.max().item<float>();
//        std::cout << "Tensor range: [" << tensor_min << ", " << tensor_max << "]" << std::endl;
//
//        // ���v���iPython���Ɣ�r�p�j
//        auto tensor_mean = inputTensor.mean().item<float>();
//        auto tensor_std = inputTensor.std().item<float>();
//        std::cout << "Tensor stats - Mean: " << tensor_mean << ", Std: " << tensor_std << std::endl;
//
//        // �ŏ��̐��l���o�́iPython���Ɗ��S��r�p�j
//        auto tensor_flat = inputTensor.flatten();
//        std::cout << "First 10 tensor values: ";
//        for (int i = 0; i < std::min(10, (int)tensor_flat.size(0)); i++) {
//            std::cout << std::fixed << std::setprecision(8) << tensor_flat[i].item<float>() << " ";
//        }
//        std::cout << std::endl;
//
//        // ����ɏڍׂȔ�r�p
//        std::cout << "First pixel (RGB): ";
//        for (int c = 0; c < 3; c++) {
//            std::cout << inputTensor[0][c][0][0].item<float>() << " ";
//        }
//        std::cout << std::endl;
//
//        // �f�o�b�O�p�摜�ۑ�
//        cv::imwrite("debug_cpp_original.png", frame);
//        cv::imwrite("debug_cpp_rgb.png", rgb);
//
//        // ���K���O�̉摜�i0-255�X�P�[���j�ŕۑ�
//        cv::Mat resized_255;
//        resized.convertTo(resized_255, CV_8U, 255.0);
//        cv::imwrite("debug_cpp_resized.png", resized_255);
//
//        saved = true;
//    }
//
//    // --- (7) GPU�]�� ---
//    inputTensor = inputTensor.to(torch::kCUDA);
//
//    // --- (8) ���_ ---
//    torch::NoGradGuard no_grad;
//    auto output = model.forward({ inputTensor }).toTensor();
//    auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU);
//
//    // �o�͓��v�i�f�o�b�O�p�j
//    std::cout << "Output tensor shape: " << output.sizes() << std::endl;
//    auto output_mean = probs.mean().item<float>();
//    auto output_std = probs.std().item<float>();
//    std::cout << "Output stats - Mean: " << output_mean << ", Std: " << output_std << std::endl;
//
//    std::cout << "First 5 prediction values: ";
//    for (int i = 0; i < std::min(5, (int)probs.numel()); i++) {
//        std::cout << std::scientific << std::setprecision(8) << probs.data_ptr<float>()[i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
//    return result;
//}

