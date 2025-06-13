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
        std::cerr << "モデルの読み込みに失敗しました: " << e.what() << std::endl;
        return false;
    }
}

static bool saved = false;  // デバッグ用フラグ

std::vector<float> predictFrame(const cv::Mat& frame, torch::jit::script::Module& model, int inputWidth, int inputHeight) {
    // --- (0) マスク適用（前処理） ---
    //cv::Mat masked;
    //cv::Mat mask = cv::imread("images/mask.png", cv::IMREAD_GRAYSCALE);
    //if (mask.size() != frame.size()) {
    //    std::cerr << "マスクサイズが入力画像と一致しません" << std::endl;
    //    // return ... or resize(mask, ...)
    //}
    //cv::bitwise_and(frame, frame, masked, mask);

    // --- (1) クロップ ---
    /*cv::Rect cropBox(330, 25, 1260, 970);
    cv::Mat cropped = masked(cropBox).clone();*/

    // --- (2) リサイズ + カラーチャンネル変換 ---
    //cv::Mat resized, rgb;
    //cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));  // e.g. 224x224

    // リサイズまでできている場合
	cv::Mat resized, rgb;
    resized = frame;

    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // --- (3) Tensor変換 ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 });  // NHWC → NCHW

    // --- (4) 標準化（ImageNet用）---
    //inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
    //inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
    //inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

	// デバッグ用に画像を保存
    if (!saved) {
        //cv::imwrite("debug_cropped.png", cropped);
        cv::imwrite("debug_resized.png", resized);
        saved = true;
    }

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (5) 推論 ---
    torch::NoGradGuard no_grad;
    auto output = model.forward({ inputTensor }).toTensor();
    auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU);

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}

