//#include <opencv2/opencv.hpp>
//#include <torch/torch.h>
//#include <torch/script.h>
//#include <iostream>
//
//int main() {
//    // OpenCV�ŃJ�����ڑ�
//    cv::VideoCapture cap(0);/*
//    if (!cap.isOpened()) {
//        std::cerr << "�J�������J���܂���ł���" << std::endl;
//        return -1;
//    }*/
//
//    // LibTorch��Tensor����
//    torch::Tensor tensor = torch::rand({ 2, 3 });
//    std::cout << "Tensor (CPU):" << std::endl << tensor << std::endl;
//
//    // CUDA�m�F
//    if (torch::cuda::is_available()) {
//        std::cout << "CUDA�͎g�p�\�ł��I" << std::endl;
//        tensor = tensor.to(torch::kCUDA);
//        std::cout << "Tensor (GPU):" << std::endl << tensor << std::endl;
//        try {
//            torch::jit::script::Module model = torch::jit::load("models/treatment_model_fold_0_best.pt");
//            model.to(torch::kCUDA);
//            model.eval();
//
//
//            std::cout << "���f���ǂݍ��ݐ���" << std::endl;
//        }
//        catch (const c10::Error& e) {
//            std::cerr << "���f���ǂݍ��ݎ��s: " << e.what() << std::endl;
//            return -1;
//        }
//    }
//    else {
//        std::cout << "CUDA�͎g�p�ł��܂���" << std::endl;
//    }
//
//    // �J�������[�v
//    //cv::Mat frame;
//    //while (true) {
//    //    cap >> frame;
//    //    if (frame.empty()) break;
//
//    //    cv::imshow("Camera Frame", frame);
//    //    if (cv::waitKey(30) == 27) break;  // ESC�ŏI��
//    //}
//
//    return 0;
//}
