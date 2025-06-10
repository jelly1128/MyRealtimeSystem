#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <iostream>

// === 設定 ===
const int FRAME_INTERVAL = 10;
const int SEQ_LEN = 100;
const int IMG_SIZE = 224;

int main() {
    // モデル読み込み
    torch::jit::script::Module treatment_model = torch::jit::load("models/treatment_model_fold_0_best.pt");
    torch::jit::script::Module organ_model = torch::jit::load("models/organ_model_v1_best.pt");
    treatment_model.to(torch::kCUDA);
    organ_model.to(torch::kCUDA);

	treatment_model.eval();
	organ_model.eval();

    // 動画ファイル読み込み
    cv::VideoCapture cap("D:/M1/動画とか/000001-002.mov");
    if (!cap.isOpened()) {
        std::cerr << "動画を開けませんでした" << std::endl;
        return -1;
    }

    int frame_idx = 0;
    std::deque<torch::Tensor> sequence_buffer;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // フレーム間引き
        if (frame_idx % FRAME_INTERVAL != 0) {
            frame_idx++;
            continue;
        }

        // 前処理（リサイズ、RGB、float正規化）
        cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);

        // OpenCV → Tensor (CHW)
        torch::Tensor img_tensor = torch::from_blob(frame.data, { IMG_SIZE, IMG_SIZE, 3 }).permute({ 2, 0, 1 }).clone();
        img_tensor = img_tensor.to(torch::kFloat32).to(torch::kCUDA);

        // === (1) 処置検出モデル推論 ===
        {
            torch::Tensor pred_treatment = treatment_model.forward({ img_tensor.unsqueeze(0) }).toTensor();
            pred_treatment = torch::sigmoid(pred_treatment).to(torch::kCPU);
            std::cout << "[Frame " << frame_idx << "] Treatment Probabilities: ";
            for (int i = 0; i < pred_treatment.size(1); ++i) {
                std::cout << std::fixed << std::setprecision(2) << pred_treatment[0][i].item<float>() * 100 << "% ";
            }
            std::cout << std::endl;
        }

        // === (2) 臓器分類用にバッファへ蓄積 ===
        sequence_buffer.push_back(img_tensor);
        if (sequence_buffer.size() > SEQ_LEN) {
            sequence_buffer.pop_front();
        }

        // === (3) バッファが100枚たまったらLSTM推論 ===
        if (sequence_buffer.size() == SEQ_LEN) {
            torch::Tensor seq_input = torch::stack(std::vector<torch::Tensor>(sequence_buffer.begin(), sequence_buffer.end())).unsqueeze(0); // (1, 100, 3, H, W)

            // LSTM初期状態
            torch::Tensor h0 = torch::zeros({ 2, 1, 128 }, torch::kFloat32).to(torch::kCUDA);
            torch::Tensor c0 = torch::zeros({ 2, 1, 128 }, torch::kFloat32).to(torch::kCUDA);

            // 推論
            auto outputs = organ_model.forward({ seq_input, h0, c0 }).toTuple();
            torch::Tensor pred = outputs->elements()[0].toTensor(); // shape: (1, 100, N_CLASS)
            torch::Tensor last = pred[0][-1];
            int label = last.argmax().item<int>();

            std::cout << "[Frame " << frame_idx << "] Organ label (last frame): " << label << std::endl;
        }

        frame_idx++;
        //cv::imshow("Frame", frame);  // 必要なら表示
        if (cv::waitKey(1) == 27) break;  // ESCで終了
    }

    return 0;
}
