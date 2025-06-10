//#include <torch/script.h>
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>
//#include <filesystem>
//
//int main() {
//    // ���f���ǂݍ���
//    torch::jit::script::Module model = torch::jit::load("models/organ_model_v1_best.pt");
//    model.to(torch::kCUDA);
//    model.eval();
//
//    // �摜�t�H���_����100�t���[���ǂݍ���
//    std::vector<cv::Mat> frames;
//    std::string img_dir = "D:/M1/����Ƃ�/20211021093634_000001-001/";
//    for (int i = 3937; i < 4037; ++i) {
//        std::string path = img_dir + std::to_string(i) + ".png";
//        cv::Mat img = cv::imread(path);
//        if (img.empty()) {
//            std::cerr << "�摜�ǂݍ��ݎ��s: " << path << std::endl;
//            return -1;
//        }
//        cv::resize(img, img, cv::Size(224, 224));
//        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
//        frames.push_back(img);
//    }
//
//    // OpenCV Mat�z�� �� Tensor��
//    torch::Tensor seq_input = torch::zeros({ 1, 100, 3, 224, 224 });
//    for (int t = 0; t < 100; ++t) {
//        torch::Tensor img_tensor = torch::from_blob(frames[t].data, { 224, 224, 3 }).permute({ 2, 0, 1 }).clone();
//        seq_input[0][t] = img_tensor;
//    }
//    seq_input = seq_input.to(torch::kFloat32).to(torch::kCUDA);
//
//    // LSTM�������
//    torch::Tensor h0 = torch::zeros({ 2, 1, 128 }, torch::kFloat32).to(torch::kCUDA);
//    torch::Tensor c0 = torch::zeros({ 2, 1, 128 }, torch::kFloat32).to(torch::kCUDA);
//
//    // ���_
//    auto outputs = model.forward({ seq_input, h0, c0 }).toTuple();
//    torch::Tensor pred = outputs->elements()[0].toTensor(); // (1, 100, N_CLASS)
//
//    // ��F�Ō�̎����̃��x����argmax�ŕ\��
//    auto last_pred = pred[0][-1]; // (N_CLASS,)
//    int label = last_pred.argmax().item<int>();
//    std::cout << "�\�����x���i�ŏI�t���[���j: " << label << std::endl;
//
//    return 0;
//}
