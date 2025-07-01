#include <iostream>
#include "config.h"
#include "src/video_loader.h"
#include "src/predictor.h"
#include "src/binarizer.h"
#include "src/result_writer.h"
#include "src/sliding_window.h"
#include "src/timeline_writer.h"

#include "src/debug.h"


int main() {
    //// 1. 初期化 
    // モデルの読み込み
    /*torch::jit::script::Module model;
    if (!loadModel(TREATMENT_MODEL_PATH, model)) {
        std::cerr << "モデルの読み込みに失敗しました。" << std::endl;
        return -1;
    }*/

    //// 2. フレーム読み込み (動画or画像)
	// 動画からフレームを読み込む
    /*std::vector<cv::Mat> frames;
    if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
        std::cerr << "動画の読み込みに失敗しました。" << std::endl;
        return -1;

    }*/

	// 画像フォルダから読み込む
	/*std::vector<cv::Mat> frames;
	if (!loadFramesFromDirectory(VIDEO_FOLDER_PATH, frames)) {
		std::cerr << "フレームの読み込みに失敗しました" << std::endl;
	} else {
		std::cout << "フレームの読み込みに成功しました。" << std::endl;
		int numFrames = frames.size();
		std::cout << "読み込んだフレーム数: " << numFrames << std::endl;
	}*/

    //// 3. 推論
	// 推論の実行
    /*std::vector<std::vector<float>> frameProbabilities;
    for (const cv::Mat& frame : frames) {
        frameProbabilities.push_back(predictFrame(frame, model, INPUT_WIDTH, INPUT_HEIGHT));
    }*/

    //// 4. 処理系
	// 推論結果のバイナリ化
    //std::vector<std::vector<int>> frameBinaryLabels = binarizeProbabilities(frameProbabilities, 0.5);

    // シーンラベルの抽出
    //std::vector<std::vector<int>> sceneClassLabels;
    //for (const std::vector<int>& vec : frameBinaryLabels) {
    //    sceneClassLabels.emplace_back(vec.begin(), vec.begin() + NUM_SCENE_CLASSES);  // 0〜5の主クラスのみ
    //}

	// スライディングウィンドウを使用してラベルを1つにまとめる
    //std::vector<int> windowedSceneLabels = slidingWindowToSingleLabel(sceneClassLabels, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, NUM_SCENE_CLASSES);

    //// 5. 出力 
    // 結果の保存 
    /*if (!saveMatrixToCSV(OUTPUT_PROBS_CSV, frameProbabilities, "prob_")) {
        std::cerr << "確率CSVの保存に失敗しました。" << std::endl;
    }
    else {
        std::cout << "推論確率を " << OUTPUT_PROBS_CSV << " に保存しました。" << std::endl;
    }*/

    /*if (!saveMatrixToCSV(OUTPUT_LABELS_CSV, frameBinaryLabels, "label_")) {
        std::cerr << "ラベルCSVの保存に失敗しました。" << std::endl;
    }
    else {
        std::cout << "ハードラベルを " << OUTPUT_LABELS_CSV << " に保存しました。" << std::endl;
    }*/

    /*if (!saveLabelsToCSV(OUTPUT_SMOOTHED_CSV, windowedSceneLabels)) {
        std::cerr << "スムーズラベルCSVの保存に失敗しました。" << std::endl;
    }
    else {
        std::cout << "スムーズラベルを " << OUTPUT_SMOOTHED_CSV << " に保存しました。" << std::endl;
	}*/

    // for debug
    std::vector<std::vector<float>> frameProbabilities;
    frameProbabilities = loadFrameProbabilitiesFromCSV(DEBUG_PROBS_CSV);

    std::vector<std::vector<int>> frameBinaryLabels;
	frameBinaryLabels = loadFrameBinariesFromCSV(DEBUG_LABELS_CSV);

    std::vector<int> windowedSceneLabels;
	windowedSceneLabels = loadWindowedSceneLabelsFromCSV(DEBUG_SMOOTHED_CSV);

    // 確率の読み取り確認用
    std::cout << "読み込んだフレーム確率数: " << frameProbabilities.size() << std::endl;
	// 確率の数量確認用
	std::cout << "読み込んだフレーム確率の列数: " << (frameProbabilities.empty() ? 0 : frameProbabilities[0].size()) << std::endl;

    // バイナリラベルの読み取り確認用
	std::cout << "読み込んだフレームバイナリラベル数: " << frameBinaryLabels.size() << std::endl;
	// バイナリラベルの数量確認用
	std::cout << "読み込んだフレームバイナリラベルの列数: " << (frameBinaryLabels.empty() ? 0 : frameBinaryLabels[0].size()) << std::endl;

	
	// スムーズラベルの読み取り確認用
	std::cout << "読み込んだスムーズラベル数: " << windowedSceneLabels.size() << std::endl;


    // タイムライン画像の出力用にcsvファイルからメインラベルを読み込む
	/*std::vector<int> mainLabels;
	if (!loadLabelsFromCSV(OUTPUT_SMOOTHED_CSV, mainLabels)) {
		std::cerr << "スムーズラベルの読み込みに失敗しました。" << std::endl;
		return -1;
	}*/

	// csv読み取り確認用
	//std::cout << "読み込んだスムーズラベル数: " << mainLabels.size() << std::endl;

    // タイムライン画像出力 
    /*if (!drawTimelineImage(mainLabels, TIMELINE_IMAGE_PATH, NUM_SCENE_CLASSES)) {
        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
    }
    else {
        std::cout << "タイムライン画像を " << TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
    }*/

	// サムネイル画像の選定
	/*std::vector<cv::Mat> thumbnails;
	if (!selectThumbnailsFromLabels(mainLabels, VIDEO_FOLDER_PATH, thumbnails, NUM_SCENE_CLASSES)) {
		std::cerr << "サムネイル画像の選定に失敗しました。" << std::endl;
		return -1;
	} else {
		std::cout << "サムネイル画像の選定に成功しました。" << std::endl;
	}*/
}
