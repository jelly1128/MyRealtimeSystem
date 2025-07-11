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
