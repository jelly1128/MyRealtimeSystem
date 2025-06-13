#include <iostream>
#include "config.h"
#include "src/video_loader.h"
#include "src/predictor.h"
#include "src/binarizer.h"
#include "src/result_writer.h"
#include "src/sliding_window.h"
#include "src/timeline_writer.h"


int main() {
	// 動画ファイルのパスとモデルのパスを設定
    /*std::vector<cv::Mat> frames;
    if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
        std::cerr << "動画の読み込みに失敗しました。" << std::endl;
        return -1;
    }*/

	// 動画ファイルの読み込み
	std::vector<cv::Mat> frames;
	if (!loadFramesFromDirectory(VIDEO_FOLDER_PATH, frames)) {
		std::cerr << "フレームの読み込みに失敗しました" << std::endl;
	} else {
		std::cout << "フレームの読み込みに成功しました。" << std::endl;
		int numFrames = frames.size();
		std::cout << "読み込んだフレーム数: " << numFrames << std::endl;
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

	// 推論結果のバイナリ化
    std::vector<std::vector<int>> hardLabels = binarizeProbabilities(allProbs, 0.5);

    // === 5. 主クラスをスライディングウィンドウで抽出 ===
    // hardLabels: [N][15] → 主クラスのみ使用
    std::vector<std::vector<int>> hardLabelsMain;
    for (const auto& vec : hardLabels) {
        hardLabelsMain.emplace_back(vec.begin(), vec.begin() + 6);  // 0〜5の主クラスのみ
    }

    std::vector<int> mainLabels = slidingWindowToSingleLabel(hardLabelsMain, 5, 1, 6);

    // === 6. 結果の保存 ===
    if (!saveMatrixToCSV(OUTPUT_PROBS_CSV, allProbs, "prob_")) {
        std::cerr << "確率CSVの保存に失敗しました。" << std::endl;
    }
    else {
        std::cout << "推論確率を " << OUTPUT_PROBS_CSV << " に保存しました。" << std::endl;
    }

    if (!saveMatrixToCSV(OUTPUT_LABELS_CSV, hardLabels, "label_")) {
        std::cerr << "ラベルCSVの保存に失敗しました。" << std::endl;
    }
    else {
        std::cout << "ハードラベルを " << OUTPUT_LABELS_CSV << " に保存しました。" << std::endl;
    }

    // === 7. タイムライン画像出力 ===
    if (!drawTimelineImage(mainLabels, TIMELINE_IMAGE_PATH)) {
        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
    }
    else {
        std::cout << "タイムライン画像を " << TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
    }
}
