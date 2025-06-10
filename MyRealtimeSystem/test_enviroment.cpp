//#include <opencv2/opencv.hpp>
//#include <torch/torch.h>
//#include <torch/script.h>
//#include <iostream>
//
//int main() {
//    // OpenCVでカメラ接続
//    cv::VideoCapture cap(0);/*
//    if (!cap.isOpened()) {
//        std::cerr << "カメラが開けませんでした" << std::endl;
//        return -1;
//    }*/
//
//    // LibTorchでTensor生成
//    torch::Tensor tensor = torch::rand({ 2, 3 });
//    std::cout << "Tensor (CPU):" << std::endl << tensor << std::endl;
//
//    // CUDA確認
//    if (torch::cuda::is_available()) {
//        std::cout << "CUDAは使用可能です！" << std::endl;
//        tensor = tensor.to(torch::kCUDA);
//        std::cout << "Tensor (GPU):" << std::endl << tensor << std::endl;
//        try {
//            torch::jit::script::Module model = torch::jit::load("models/treatment_model_fold_0_best.pt");
//            model.to(torch::kCUDA);
//            model.eval();
//
//
//            std::cout << "モデル読み込み成功" << std::endl;
//        }
//        catch (const c10::Error& e) {
//            std::cerr << "モデル読み込み失敗: " << e.what() << std::endl;
//            return -1;
//        }
//    }
//    else {
//        std::cout << "CUDAは使用できません" << std::endl;
//    }
//
//    // カメラループ
//    //cv::Mat frame;
//    //while (true) {
//    //    cap >> frame;
//    //    if (frame.empty()) break;
//
//    //    cv::imshow("Camera Frame", frame);
//    //    if (cv::waitKey(30) == 27) break;  // ESCで終了
//    //}
//
//    return 0;
//}
