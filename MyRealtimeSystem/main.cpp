#include <iostream>
#include "config.h"
#include "src/video_loader.h"
#include "src/predictor.h"
#include "src/result_writer.h"

int main() {
	// 動画ファイルのパスとモデルのパスを設定
    std::vector<cv::Mat> frames;
    if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
        std::cerr << "動画の読み込みに失敗しました。" << std::endl;
        return -1;
    }

	// モデルの読み込み
    torch::jit::script::Module model;
    if (!loadModel(TREATMENT_MODEL_PATH, model)) {
        std::cerr << "モデルの読み込みに失敗しました。" << std::endl;
        return -1;
    }

	// 推論の実行
    std::vector<std::vector<float>> allProbs;
    for (const auto& frame : frames) {
        allProbs.push_back(predictFrame(frame, model, INPUT_WIDTH, INPUT_HEIGHT));
    }

	// 結果のCSVファイルへの保存
    if (!saveProbabilitiesToCSV(OUTPUT_CSV, allProbs)) {
        std::cerr << "CSVファイルの保存に失敗しました。" << std::endl;
        return -1;
    }

    //std::cout << "処理完了。結果を " << OUTPUT_CSV << " に保存しました。" << std::endl;
    return 0;
}
