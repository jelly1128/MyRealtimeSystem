//#include <torch/script.h>
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//int main() {
//    // モデルの読み込み
//    torch::jit::script::Module model = torch::jit::load("models/treatment_model_fold_0_best.pt");
//    model.to(torch::kCUDA);
//    model.eval();
//
//    // 入力画像の読み込み
//    cv::Mat img = cv::imread("images/sample.png");
//    if (img.empty()) {
//        std::cerr << "画像が読み込めませんでした。" << std::endl;
//        return -1;
//    }
//
//    // 前処理
//    cv::resize(img, img, cv::Size(224, 224));
//    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
//    torch::Tensor input = torch::from_blob(img.data, { 1, 224, 224, 3 });
//    input = input.permute({ 0, 3, 1, 2 }).to(torch::kFloat32).to(torch::kCUDA);
//
//    // 推論とsigmoid適用
//    torch::Tensor output = model.forward({ input }).toTensor();
//    torch::Tensor probs = torch::sigmoid(output).to(torch::kCPU);
//
//    // 結果表示
//    std::vector<float> probabilities(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
//    for (size_t i = 0; i < probabilities.size(); ++i) {
//        std::cout << "Class " << i << ": " << probabilities[i] * 100.0f << " %" << std::endl;
//    }
//
//    return 0;
//}
