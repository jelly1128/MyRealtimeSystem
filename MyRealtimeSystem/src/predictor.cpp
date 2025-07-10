#include "predictor.h"
#include <torch/torch.h>
#include <iomanip>  // std::setprecision, std::fixed, std::scientific用

bool loadModel(const std::string& modelPath, torch::jit::script::Module& model) {
    try {
        model = torch::jit::load(modelPath);
        model.to(torch::kCUDA);
        model.eval();
        return true;
    }
    catch (const c10::Error& e) {
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
    //cv::Rect cropBox(330, 25, 1260, 970);
    //cv::Mat cropped = masked(cropBox).clone();

    //// --- (2) リサイズ + カラーチャンネル変換 ---
    //cv::Mat resized, rgb;
    //cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));  // e.g. 224x224

    // リサイズまでできている場合
	cv::Mat resized, rgb;
    resized = frame.clone();

    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    if (!rgb.isContinuous()) {
        rgb = rgb.clone();  // from_blobの前に連続化
    }

    // --- (3) Tensor変換 ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC → NCHW

	// デバッグ用に画像を保存
    /*if (!saved) {
		cv::imwrite("debug_frame.png", frame);
		//cv::imwrite("debug_masked.png", masked);
		cv::imwrite("debug_rgb.png", rgb);
        //cv::imwrite("debug_cropped.png", cropped);
        cv::imwrite("debug_resized.png", resized);
        saved = true;
    }*/

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (5) 推論 ---
	torch::NoGradGuard no_grad;                                    // 勾配計算を無効化
	auto output = model.forward({ inputTensor }).toTensor();       // モデルに入力を渡す
	auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU); // シグモイド関数を適用し、CPUに戻す

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}