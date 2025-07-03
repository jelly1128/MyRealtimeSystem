#include <iostream>
#include "config.h"
#include "src/video_loader.h"
#include "src/predictor.h"
#include "src/binarizer.h"
#include "src/result_writer.h"
#include "src/sliding_window.h"
#include "src/timeline_writer.h"

#include "src/debug.h"


struct FrameData {
    int frameIndex;
    std::vector<float> probabilities;
    int swLabel;
    float S_target;
    float S_event;
};


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

    // タイムライン画像出力
    /*if (!drawTimelineImage(windowedSceneLabels, TIMELINE_IMAGE_PATH, NUM_SCENE_CLASSES)) {
        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
    }
    else {
        std::cout << "タイムライン画像を " << TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
    }*/

    // for debug
    std::vector<std::vector<float>> frameProbabilities;
    frameProbabilities = loadFrameProbabilitiesFromCSV(DEBUG_PROBS_CSV);

    std::vector<std::vector<int>> frameBinaryLabels;
	frameBinaryLabels = loadFrameBinariesFromCSV(DEBUG_LABELS_CSV);

    std::vector<int> windowedSceneLabels;
	windowedSceneLabels = loadWindowedSceneLabelsFromCSV(DEBUG_SMOOTHED_CSV);

    if (!drawTimelineImage(windowedSceneLabels, TIMELINE_IMAGE_PATH, NUM_SCENE_CLASSES)) {
        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
    }
    else {
        std::cout << "タイムライン画像を " << TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
    }

    // 画像ごとのデータをまとめた構造体
	std::vector<FrameData> frameDataList;
	int halfWindowSize = SLIDING_WINDOW_SIZE / 2;
	std::cout << "ハーフウィンドウサイズ: " << halfWindowSize << std::endl;
	std::cout << "フレーム数: " << frameProbabilities.size() << std::endl;
	std::cout << "ウィンドウ適用後のラベル数: " << windowedSceneLabels.size() << std::endl;

	// 各フレームのデータを構造体に格納
    for (size_t i = 0; i < frameProbabilities.size(); ++i) {
        FrameData data;
        data.frameIndex = i;
        data.probabilities = frameProbabilities[i];

        if (i >= halfWindowSize && i < frameProbabilities.size() - halfWindowSize) {
            int labelIndex = i - halfWindowSize;
            data.swLabel = windowedSceneLabels[labelIndex];

            if (data.swLabel >= 0 && data.swLabel < data.probabilities.size()) {
                data.S_target = data.probabilities[data.swLabel];
                data.S_event = 0.0f;
                for (int j = 7; j <= 14; ++j) {
                    data.S_event += data.probabilities[j];
                }
            }
        }
        else {
            data.swLabel = -1;  // ラベル無し領域
            data.S_target = 0.0f;
            data.S_event = 0.0f;
        }

        frameDataList.push_back(data);
    }

    // デバッグ用にフレームデータを表示
    /*for (const FrameData& data : frameDataList) {
        std::cout << "Frame " << data.frameIndex << ": SW Label = " << data.swLabel << ", Probabilities = ";
        for (float prob : data.probabilities) {
            std::cout << prob << " ";
		}
		std::cout << ", S_target = " << data.S_target << ", S_event = " << data.S_event;
        std::cout << std::endl;
	}*/

    // --- ラベルごとのメイン区間を抽出する ---
    std::map<int, std::pair<int, int>> longestSegments;  // label → {startIdx, endIdx}
    std::map<int, int> currentStart;
    std::map<int, int> currentLength;
    std::map<int, int> maxLength;

    int prevLabel = -2;

    for (size_t i = 0; i < frameDataList.size(); ++i) {
	    int label = frameDataList[i].swLabel;
	    if (label == -1) continue;

	    if (label != prevLabel) {
		    // ラベルが変化したので初期化
		    currentStart[label] = static_cast<int>(i);
		    currentLength[label] = 1;
	    } else {
		    currentLength[label]++;
	    }

	    // 長さが更新された場合のみ記録
	    if (currentLength[label] > maxLength[label]) {
		    maxLength[label] = currentLength[label];
		    longestSegments[label] = {currentStart[label], static_cast<int>(i)};
	    }

	    prevLabel = label;
    }

    // --- 確認出力 ---
    std::cout << "各ラベルのメイン区間:" << std::endl;
    for (const auto& [label, range] : longestSegments) {
	    std::cout << "Label " << label << " → ["
		    << range.first << ", " << range.second << "] (length = "
		    << range.second - range.first + 1 << ")" << std::endl;
    }


	// サムネイル画像の選定
	/*std::vector<cv::Mat> thumbnails;
	if (!selectThumbnailsFromLabels(windowedSceneLabels, VIDEO_FOLDER_PATH, thumbnails, NUM_SCENE_CLASSES)) {
		std::cerr << "サムネイル画像の選定に失敗しました。" << std::endl;
		return -1;
	} else {
		std::cout << "サムネイル画像の選定に成功しました。" << std::endl;
	}*/

    // サムネイル画像の保存
    /*for (size_t i = 0; i < thumbnails.size(); ++i) {
        std::string thumbnailPath = "outputs/thumbnail_" + std::to_string(i) + ".png";
        if (!cv::imwrite(thumbnailPath, thumbnails[i])) {
            std::cerr << "サムネイル画像の保存に失敗しました: " << thumbnailPath << std::endl;
        } else {
            std::cout << "サムネイル画像を保存しました: " << thumbnailPath << std::endl;
        }
    }*/
	return 0;
}
