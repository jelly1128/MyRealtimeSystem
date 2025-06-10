//#include <torch/script.h>
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>
//#include <filesystem>
//
//int main() {
//    // モデル読み込み
//    torch::jit::script::Module model = torch::jit::load("models/organ_model_v1_best.pt");
//    model.to(torch::kCUDA);
//    model.eval();
//
//    // 画像フォルダから100フレーム読み込む
//    std::vector<cv::Mat> frames;
//    std::string img_dir = "D:/M1/動画とか/20211021093634_000001-001/";
//    for (int i = 3937; i < 4037; ++i) {
//        std::string path = img_dir + std::to_string(i) + ".png";
//        cv::Mat img = cv::imread(path);
//        if (img.empty()) {
//            std::cerr << "画像読み込み失敗: " << path << std::endl;
//            return -1;
//        }
//        cv::resize(img, img, cv::Size(224, 224));
//        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
//        frames.push_back(img);
//    }
//
//    // OpenCV Mat配列 → Tensor化
//    torch::Tensor seq_input = torch::zeros({ 1, 100, 3, 224, 224 });
//    for (int t = 0; t < 100; ++t) {
//        torch::Tensor img_tensor = torch::from_blob(frames[t].data, { 224, 224, 3 }).permute({ 2, 0, 1 }).clone();
//        seq_input[0][t] = img_tensor;
//    }
//    seq_input = seq_input.to(torch::kFloat32).to(torch::kCUDA);
//
//    // LSTM初期状態
//    torch::Tensor h0 = torch::zeros({ 2, 1, 128 }, torch::kFloat32).to(torch::kCUDA);
//    torch::Tensor c0 = torch::zeros({ 2, 1, 128 }, torch::kFloat32).to(torch::kCUDA);
//
//    // 推論
//    auto outputs = model.forward({ seq_input, h0, c0 }).toTuple();
//    torch::Tensor pred = outputs->elements()[0].toTensor(); // (1, 100, N_CLASS)
//
//    // 例：最後の時刻のラベルをargmaxで表示
//    auto last_pred = pred[0][-1]; // (N_CLASS,)
//    int label = last_pred.argmax().item<int>();
//    std::cout << "予測ラベル（最終フレーム）: " << label << std::endl;
//
//    return 0;
//}
