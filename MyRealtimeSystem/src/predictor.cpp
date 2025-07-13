#include "predictor.h"
#include <torch/torch.h>


// モデルを読み込む関数
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


// 処置検出モデルの推論を実行する関数
std::vector<float> runTreatmentInference(
    const torch::Tensor& frameTensor,
    torch::jit::script::Module& treatmentModel
) {
    torch::NoGradGuard no_grad;                                    // 勾配計算を無効化
    torch::Tensor input = frameTensor.to(torch::kCUDA);            // GPU転送
    auto output = treatmentModel.forward({ input }).toTensor();    // モデルに入力を渡す
	auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU); // シグモイド関数を適用し、CPUに戻す

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}


// 臓器分類モデルの推論を実行する関数
int runOrganInference(
    const torch::Tensor& frameTensor, 
    torch::jit::script::Module& organModel,
    torch::Tensor& h_0, torch::Tensor& c_0
) {
    torch::NoGradGuard no_grad;

    // [1, 3, H, W] → [1, 1, 3, H, W]
    torch::Tensor input = frameTensor.unsqueeze(0);

    // モデルのforward
    auto outputs = organModel.forward({ input.to(torch::kCUDA), h_0, c_0}).toTuple();
    torch::Tensor output = outputs->elements()[0].toTensor()[0]; // 推論出力
    h_0 = outputs->elements()[1].toTensor(); // 新しい隠れ状態
    c_0 = outputs->elements()[2].toTensor(); // 新しいセル状態

    // 最大要素のインデックス（整数ラベル）を取得
    int label = output.argmax(1)[0].item<int>(); // shape: [1], [0]でint化

    return label;
}