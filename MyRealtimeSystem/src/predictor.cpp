#include "predictor.h"
#include <torch/torch.h>

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

std::vector<float> predictFrame(const cv::Mat& frame, torch::jit::script::Module& model, int inputWidth, int inputHeight) {
    // --- (1) �N���b�v ---
    cv::Rect cropBox(330, 25, 1260, 970);
    cv::Mat cropped = frame(cropBox).clone();

    // --- (2) ���T�C�Y + �J���[�`�����l���ϊ� ---
    cv::Mat resized, rgb;
    cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));  // e.g. 224x224
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // --- (3) Tensor�ϊ� ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 });  // NHWC �� NCHW

    // --- (4) �W�����iImageNet�p�j---
    //inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
    //inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
    //inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (5) ���_ ---
    torch::NoGradGuard no_grad;
    auto output = model.forward({ inputTensor }).toTensor();
    auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU);

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}

