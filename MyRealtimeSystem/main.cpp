#include "config/config.h"
#include "src/video_loader.h"
#include "src/predictor.h"
#include "src/binarizer.h"
#include "src/result_writer.h"
#include "src/sliding_window.h"
#include "src/timeline_writer.h"
#include "src/thumbnail.h"

#include "src/debug.h"

int main() {
	// ログの初期化
	initLog(DEBUG_LOG_FILE_PATH);
	log("プログラム開始", true);
	TimeLogger timerAll("全体処理時間");

	TimeLogger timerLoad("モデル読み込み");

	// 1. 初期化
    // モデルの読み込み
    torch::jit::script::Module treatmentModel, organModel;
    if (!loadModel(TREATMENT_MODEL_PATH, treatmentModel) ||
		!loadModel(ORGAN_MODEL_PATH, organModel)) {
		log("モデルの読み込みに失敗しました。", true);
		closeLog();
        return -1;
    }

	timerLoad.stop();

	//TimeLogger timerRead("フレーム読み込み");

	// 2. フレーム読み込み (動画or画像)
	// 動画からフレームを読み込む
 //   std::vector<cv::Mat> frames;
 //   if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
	//	log("動画のフレームの読み込みに失敗しました。", true);
	//	closeLog();
 //       return -1;
	//} else {
	//	log("動画のフレームの読み込みに成功しました。", true);
	//	//showFrames(frames);  // フレームを表示する関数を呼び出す（デバッグ）
	//}

	//timerRead.stop();

	//TimeLogger timerPreprocess("フレーム前処理");

	// 画像前処理
	/*std::vector<torch::Tensor> frameTreatmentTensors;
	for (cv::Mat& frame : frames) {
		torch::Tensor frameTensor = preprocessFrameForTreatment(frame, INPUT_WIDTH, INPUT_HEIGHT, TREATMENT_CROP_BOX, cv::imread(MASK_IMAGE_PATH, cv::IMREAD_GRAYSCALE));
		frameTreatmentTensors.push_back(frameTensor);
	}*/

	/*std::vector<torch::Tensor> frameOrganTensors;
	for (cv::Mat& frame : frames) {
		torch::Tensor frameTensor = preprocessFrameForOrgan(frame, INPUT_WIDTH, INPUT_HEIGHT, ORGAN_INPUT_WIDTH, ORGAN_CROP_BOX, cv::imread(MASK_IMAGE_PATH, cv::IMREAD_GRAYSCALE));
		frameOrganTensors.push_back(frameTensor);
	}*/

	TimeLogger timerRead("フレーム読み込み");

	// 画像フォルダから読み込む
	std::vector<cv::Mat> frames;
	if (!loadFramesFromDirectory(VIDEO_FOLDER_PATH, frames)) {
		log("フレームの読み込みに失敗しました。", true);
		closeLog();
	} else {
		log("フレームの読み込みに成功しました。", true);
		log("読み込んだフレーム数: " + std::to_string(frames.size()), true);
		//showFrames(frames);  // フレームを表示する関数を呼び出す（デバッグ）
	}

	timerRead.stop();

	TimeLogger timerPreprocess("フレーム前処理");

	// 画像前処理
	std::vector<torch::Tensor> frameTreatmentTensors;
	for (cv::Mat& frame : frames) {
		torch::Tensor frameTensor = preprocessFrameForTreatment(frame, INPUT_WIDTH, INPUT_HEIGHT);
		frameTreatmentTensors.push_back(frameTensor);
	}

	std::vector<torch::Tensor> frameOrganTensors;
	for (cv::Mat& frame : frames) {
		torch::Tensor frameTensor = preprocessFrameForOrgan(frame, INPUT_WIDTH, INPUT_HEIGHT);
		frameOrganTensors.push_back(frameTensor);
	}

	timerPreprocess.stop();

	TimeLogger timerInference("処置検出の推論");

	// 3. 推論
	// 処置検出の推論の実行
    /*std::vector<std::vector<float>> treatmentProbabilities;
    for (const torch::Tensor& frameTensor : frameTreatmentTensors) {
		treatmentProbabilities.push_back(runTreatmentInference(frameTensor, treatmentModel));
	}*/

	// 隠れ状態とセル状態の初期化
	//torch::Tensor h_0 = torch::zeros({ 2, 1, 128 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
	//torch::Tensor c_0 = torch::zeros({ 2, 1, 128 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

	//// 臓器分類の推論の実行（まだ実装できてない）
	//std::vector<int> organLabels;
	//for (const torch::Tensor& frameTensor : frameOrganTensors) {
	//	// 推論部分やfrom_blobの周辺
	//	int label = runOrganInference(frameTensor, organModel, h_0, c_0);
	//	organLabels.push_back(label);
	//}


	// 推論結果の保存
	/*if (!saveMatrixToCSV(TREATMENT_OUTPUT_PROBS_CSV, treatmentProbabilities, "prob_")) {
		log("確率CSVの保存に失敗しました。", true);
		closeLog();
		return -1;
	} else {
		log("推論確率を " + TREATMENT_OUTPUT_PROBS_CSV + " に保存しました。", true);
	}

	if (!saveLabelsToCSV(ORGAN_OUTPUT_LABELS_CSV, organLabels)) {
		log("臓器分類の確率CSVの保存に失敗しました。", true);
		closeLog();
		return -1;
	}
	else {
		log("臓器分類の推論確率を " + ORGAN_OUTPUT_LABELS_CSV + " に保存しました。", true);
	}*/

	timerInference.stop();

    // 4. 処理系
	// 推論結果のバイナリ化
    //std::vector<std::vector<int>> frameBinaryLabels = binarizeProbabilities(frameProbabilities, 0.5);

    // シーンラベルの抽出
    //std::vector<std::vector<int>> sceneClassLabels;
    //for (const std::vector<int>& vec : frameBinaryLabels) {
    //    sceneClassLabels.emplace_back(vec.begin(), vec.begin() + NUM_SCENE_CLASSES);  // 0〜5の主クラスのみ
    //}
	timerAll.stop();
	closeLog();
    return 0;
}

//int main() {
//	// ログ出力用のストリーム
//    std::stringstream logStream;
//
//    //// 1. 初期化 
//    // モデルの読み込み
//    torch::jit::script::Module model;
//    if (!loadModel(TREATMENT_MODEL_PATH, model)) {
//        std::cerr << "モデルの読み込みに失敗しました。" << std::endl;
//        return -1;
//    }
//
//    //// 2. フレーム読み込み (動画or画像)
//	// 動画からフレームを読み込む
//    /*std::vector<cv::Mat> frames;
//    if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
//        std::cerr << "動画の読み込みに失敗しました。" << std::endl;
//        return -1;
//
//    }*/
//
//	// 画像フォルダから読み込む
//	std::vector<cv::Mat> frames;
//	if (!loadFramesFromDirectory(VIDEO_FOLDER_PATH, frames)) {
//		std::cerr << "フレームの読み込みに失敗しました" << std::endl;
//	} else {
//		std::cout << "フレームの読み込みに成功しました。" << std::endl;
//		int numFrames = frames.size();
//		std::cout << "読み込んだフレーム数: " << numFrames << std::endl;
//	}
//
//    //// 3. 推論
//	// 推論の実行
//    std::vector<std::vector<float>> frameProbabilities;
//    for (const cv::Mat& frame : frames) {
//        frameProbabilities.push_back(predictFrame(frame, model, INPUT_WIDTH, INPUT_HEIGHT));
//    }
//
//    //// 4. 処理系
//	// 推論結果のバイナリ化
//    std::vector<std::vector<int>> frameBinaryLabels = binarizeProbabilities(frameProbabilities, 0.5);
//
//    // シーンラベルの抽出
//    std::vector<std::vector<int>> sceneClassLabels;
//    for (const std::vector<int>& vec : frameBinaryLabels) {
//        sceneClassLabels.emplace_back(vec.begin(), vec.begin() + NUM_SCENE_CLASSES);  // 0〜5の主クラスのみ
//    }
//
//	// スライディングウィンドウを使用してラベルを1つにまとめる
//    std::vector<int> windowedSceneLabels = slidingWindowToSingleLabel(sceneClassLabels, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, NUM_SCENE_CLASSES);
//
//    //// 5. 出力 
//    // 結果の保存 
//    if (!saveMatrixToCSV(OUTPUT_PROBS_CSV, frameProbabilities, "prob_")) {
//        std::cerr << "確率CSVの保存に失敗しました。" << std::endl;
//    }
//    else {
//        std::cout << "推論確率を " << OUTPUT_PROBS_CSV << " に保存しました。" << std::endl;
//    }
//
//    if (!saveMatrixToCSV(OUTPUT_LABELS_CSV, frameBinaryLabels, "label_")) {
//        std::cerr << "ラベルCSVの保存に失敗しました。" << std::endl;
//    }
//    else {
//        std::cout << "ハードラベルを " << OUTPUT_LABELS_CSV << " に保存しました。" << std::endl;
//    }
//
//    if (!saveLabelsToCSV(OUTPUT_SMOOTHED_CSV, windowedSceneLabels)) {
//        std::cerr << "スムーズラベルCSVの保存に失敗しました。" << std::endl;
//    }
//    else {
//        std::cout << "スムーズラベルを " << OUTPUT_SMOOTHED_CSV << " に保存しました。" << std::endl;
//	}
//
//    // タイムライン画像出力
//    if (!drawTimelineImage(windowedSceneLabels, TIMELINE_IMAGE_PATH, NUM_SCENE_CLASSES)) {
//        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
//    }
//    else {
//        std::cout << "タイムライン画像を " << TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
//    }
//
//    // for debug
//    /*std::vector<std::vector<float>> frameProbabilities;
//    frameProbabilities = loadFrameProbabilitiesFromCSV(DEBUG_PROBS_CSV);
//
//    std::vector<std::vector<int>> frameBinaryLabels;
//	frameBinaryLabels = loadFrameBinariesFromCSV(DEBUG_LABELS_CSV);
//
//    std::vector<int> windowedSceneLabels;
//	windowedSceneLabels = loadWindowedSceneLabelsFromCSV(DEBUG_SMOOTHED_CSV);*/
//
//    // 推論結果の読み込みからの場合
//    /*std::vector<std::vector<float>> frameProbabilities;
//    frameProbabilities = loadFrameProbabilitiesFromCSV(OUTPUT_PROBS_CSV);
//
//    std::vector<std::vector<int>> frameBinaryLabels;
//    frameBinaryLabels = loadFrameBinariesFromCSV(OUTPUT_LABELS_CSV);
//
//    std::vector<int> windowedSceneLabels;
//    windowedSceneLabels = loadWindowedSceneLabelsFromCSV(OUTPUT_SMOOTHED_CSV);*/
//
//	// タイムライン画像の出力
//    /*if (!drawTimelineImage(windowedSceneLabels, TIMELINE_IMAGE_PATH, NUM_SCENE_CLASSES)) {
//        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
//    }
//    else {
//        std::cout << "タイムライン画像を " << TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
//    }*/
//
//
//    std::vector<std::filesystem::directory_entry> imageEntries;
//    // 元のファイルを取得
//    for (const auto& entry : std::filesystem::directory_iterator(VIDEO_FOLDER_PATH)) {
//        if (entry.is_regular_file()) {
//            imageEntries.push_back(entry);
//        }
//    }
//
//    // ファイル名末尾の数字を抽出してソート
//    std::sort(imageEntries.begin(), imageEntries.end(), [](const auto& a, const auto& b) {
//        auto getFrameNumber = [](const std::string& filename) -> int {
//            size_t lastUnderscore = filename.rfind('_');
//            size_t dot = filename.rfind('.');
//            if (lastUnderscore == std::string::npos || dot == std::string::npos) return -1;
//            std::string numStr = filename.substr(lastUnderscore + 1, dot - lastUnderscore - 1);
//            return std::stoi(numStr);
//        };
//
//    return getFrameNumber(a.path().filename().string()) < getFrameNumber(b.path().filename().string());
//    });
//
//    // ソート結果をフレーム順のパスベクトルへ変換
//    std::vector<std::string> frameImagePaths;
//    std::map<int, std::string> frameIndexToFileName;
//
//    for (size_t i = 0; i < imageEntries.size(); ++i) {
//        std::string filename = imageEntries[i].path().filename().string();
//        frameImagePaths.push_back(filename);
//        frameIndexToFileName[i] = filename;  // frameIndexとしてiを割り当て
//    }
//
//
//    // 画像ごとのデータをまとめた構造体
//	std::vector<FrameData> frameDataList;
//	int halfWindowSize = SLIDING_WINDOW_SIZE / 2;
//	std::cout << "ハーフウィンドウサイズ: " << halfWindowSize << std::endl;
//	std::cout << "フレーム数: " << frameProbabilities.size() << std::endl;
//	std::cout << "ウィンドウ適用後のラベル数: " << windowedSceneLabels.size() << std::endl;
//
//    logStream << "ハーフウィンドウサイズ: " << halfWindowSize << std::endl;
//    logStream << "フレーム数: " << frameProbabilities.size() << std::endl;
//    logStream << "ウィンドウ適用後のラベル数: " << windowedSceneLabels.size() << std::endl;
//
//	// 各フレームのデータを構造体に格納
//    for (size_t i = 0; i < frameProbabilities.size(); ++i) {
//        FrameData data;
//        data.frameIndex = i;
//        data.probabilities = frameProbabilities[i];
//
//        if (i >= halfWindowSize && i < frameProbabilities.size() - halfWindowSize) {
//            int labelIndex = i - halfWindowSize;
//            data.swLabel = windowedSceneLabels[labelIndex];
//
//            if (data.swLabel >= 0 && data.swLabel < data.probabilities.size()) {
//                data.S_target = data.probabilities[data.swLabel];
//                data.S_event = 0.0f;
//                for (int j = 7; j <= 14; ++j) {
//                    data.S_event += data.probabilities[j];
//                }
//            }
//        }
//        else {
//            data.swLabel = -1;  // ラベル無し領域
//            data.S_target = 0.0f;
//            data.S_event = 0.0f;
//        }
//
//        frameDataList.push_back(data);
//    }
//
//    // デバッグ用にフレームデータと画像パスを表示
// //   for (const FrameData& data : frameDataList) {
//	//	// 画像ファイル名を取得
// //       std::string fileName = frameIndexToFileName.count(data.frameIndex)
// //           ? frameIndexToFileName.at(data.frameIndex)
//	//		: "unknown";
//	//	// 画像ファイル名とフレームデータを表示
//	//	std::cout << "File: " << fileName << " (Frame " << data.frameIndex << ") ";
// //       std::cout << ": SW Label = " << data.swLabel << ", Probabilities = ";
// //       for (float prob : data.probabilities) {
// //           std::cout << prob << " ";
//	//	}
//	//	std::cout << ", S_target = " << data.S_target << ", S_event = " << data.S_event;
// //       std::cout << std::endl;
//	//}
//
//    // --- ラベルごとのメイン区間を抽出する ---
//    std::map<int, std::pair<int, int>> longestSegments;  // label → {startIdx, endIdx}
//    std::map<int, int> currentStart;
//    std::map<int, int> currentLength;
//    std::map<int, int> maxLength;
//
//    int prevLabel = -2;
//
//    for (size_t i = 0; i < frameDataList.size(); ++i) {
//	    int label = frameDataList[i].swLabel;
//	    if (label == -1) continue;
//
//	    if (label != prevLabel) {
//		    // ラベルが変化したので初期化
//		    currentStart[label] = static_cast<int>(i);
//		    currentLength[label] = 1;
//	    } else {
//		    currentLength[label]++;
//	    }
//
//	    // 長さが更新された場合のみ記録
//	    if (currentLength[label] > maxLength[label]) {
//		    maxLength[label] = currentLength[label];
//		    longestSegments[label] = {currentStart[label], static_cast<int>(i)};
//	    }
//
//	    prevLabel = label;
//    }
//
//    // --- 確認出力 ---
//    std::cout << "各ラベルのメイン区間:" << std::endl;
//    for (const auto& [label, range] : longestSegments) {
//	    std::cout << "Label " << label << " → ["
//		    << range.first << ", " << range.second << "] (length = "
//		    << range.second - range.first + 1 << ")" << std::endl;
//    }
//
//    logStream << "各ラベルのメイン区間:\n";
//    for (const auto& [label, range] : longestSegments) {
//        logStream << "Label " << label << " → [" << range.first << ", " << range.second
//            << "] (length = " << (range.second - range.first + 1) << ")\n";
//    }
//
//    // --- フェーズ②：各ラベルのメイン区間から上位スコアフレームを選定 ---
//    const int TOP_K = 20;  // 上位何枚を取得するか
//
//    std::map<int, std::vector<FrameData>> topFramesPerLabel;  // label → 上位3フレーム
//
//    for (const auto& [label, range] : longestSegments) {
//        int start = range.first;
//        int end = range.second;
//
//        std::vector<FrameData> candidates;
//        for (int i = start; i <= end; ++i) {
//            if (frameDataList[i].swLabel == label) {
//                candidates.push_back(frameDataList[i]);
//            }
//        }
//
//        const int FRAME_GAP = 15;  // 最低フレーム間隔（3fps換算で5秒）
//
//        // スコア順に並べる
//        std::sort(candidates.begin(), candidates.end(), [](const FrameData& a, const FrameData& b) {
//            return (a.S_target - a.S_event) > (b.S_target - b.S_event);
//            });
//
//        // FRAME_GAPを考慮して間隔をあけながらTOP_K件選定
//        std::vector<FrameData> selected;
//        std::set<int> usedIndices;
//
//        for (const auto& cand : candidates) {
//            bool tooClose = false;
//            for (int used : usedIndices) {
//                if (std::abs(cand.frameIndex - used) < FRAME_GAP) {
//                    tooClose = true;
//                    break;
//                }
//            }
//            if (!tooClose) {
//                selected.push_back(cand);
//                usedIndices.insert(cand.frameIndex);
//                if (selected.size() >= TOP_K) break;
//            }
//        }
//
//        topFramesPerLabel[label] = selected;
//
//    }
//
//    //std::cout << "\n--- 各ラベルのサムネイル候補（上位 " << TOP_K << " 件） ---\n";
//    for (const auto& [label, frames] : topFramesPerLabel) {
//        std::cout << "Label " << label << "：" << std::endl;
//        for (const auto& f : frames) {
//            std::string fileName = frameIndexToFileName.count(f.frameIndex)
//                ? frameIndexToFileName.at(f.frameIndex)
//                : "unknown";
//
//            std::cout << "  File: " << fileName
//                << " (Frame " << f.frameIndex << ")"
//                << " | S_target: " << f.S_target
//                << " | S_event: " << f.S_event
//                << " | Score: " << (f.S_target - f.S_event)
//                << std::endl;
//        }
//    }
//
//
//
//    // === グリッド描画 ===
//    const int thumbRows = 4, thumbCols = 5;
//    const int thumbWidth = 224, thumbHeight = 224;
//    for (const auto& [label, frames] : topFramesPerLabel) {
//        cv::Mat canvas(thumbRows * thumbHeight, thumbCols * thumbWidth, CV_8UC3, cv::Scalar(255, 255, 255));
//        int count = 0;
//        for (const auto& f : frames) {
//            if (!frameIndexToFileName.count(f.frameIndex)) continue;
//            std::string fileName = frameIndexToFileName[f.frameIndex];
//            std::string imgPath = VIDEO_FOLDER_PATH + "/" + fileName;
//
//            cv::Mat img = cv::imread(imgPath);
//            if (img.empty()) continue;
//            cv::resize(img, img, cv::Size(thumbWidth, thumbHeight));
//
//            int row = count / thumbCols, col = count % thumbCols;
//            cv::Rect roi(col * thumbWidth, row * thumbHeight, thumbWidth, thumbHeight);
//            img.copyTo(canvas(roi));
//
//            count++;
//            if (count >= TOP_K) break;
//        }
//        std::string outPath = THUMBNAIL_PATH + "_thumbnail_grid_label" + std::to_string(label) + ".png";
//        cv::imwrite(outPath, canvas);
//        std::cout << "保存完了: " << outPath << std::endl;
//    }
//
//
//    // --- フェーズ③：高周波エネルギーに基づくサムネイル選定 ---
//    std::map<int, std::vector<std::pair<int, double>>> highFreqTopFramesPerLabel;  // label → (frameIndex, energy)
//
//    for (const auto& [label, range] : longestSegments) {
//        int start = range.first;
//        int end = range.second;
//
//        std::vector<std::pair<int, double>> scoredFrames;
//
//        for (int i = start; i <= end; ++i) {
//            if (frameDataList[i].swLabel != label) continue;
//
//            // 対応する画像ファイルパスを取得
//            if (!frameIndexToFileName.count(i)) continue;
//            std::string fileName = frameIndexToFileName.at(i);
//            std::string imgPath = VIDEO_FOLDER_PATH + "/" + fileName;
//
//            cv::Mat img = cv::imread(imgPath);
//            if (img.empty()) continue;
//
//            double energy = computeHighFrequencyEnergy(img);
//            scoredFrames.emplace_back(i, energy);
//        }
//
//        // 高周波スコアで降順ソート
//        // --- 間隔付き選定ロジックここから ---
//        const int FRAME_GAP = 15;  // フレーム間の最小間隔
//
//        std::vector<std::pair<int, double>> selected;
//        std::set<int> usedIndicesForHF;
//
//        std::sort(scoredFrames.begin(), scoredFrames.end(),
//            [](const auto& a, const auto& b) { return a.second > b.second; });
//
//        for (const auto& [frameIndex, energy] : scoredFrames) {
//            bool tooClose = false;
//            for (int used : usedIndicesForHF) {
//                if (std::abs(frameIndex - used) < FRAME_GAP) {
//                    tooClose = true;
//                    break;
//                }
//            }
//
//            if (!tooClose) {
//                selected.emplace_back(frameIndex, energy);
//                usedIndicesForHF.insert(frameIndex);
//                if (selected.size() >= TOP_K) break;
//            }
//        }
//
//        highFreqTopFramesPerLabel[label] = selected;
//    }
//
//    // --- 出力確認 ---
//    std::cout << "\n--- 高周波エネルギーベースのサムネイル候補（上位 " << TOP_K << " 件） ---\n";
//    for (const auto& [label, frames] : highFreqTopFramesPerLabel) {
//        std::cout << "Label " << label << "：" << std::endl;
//        for (const auto& [frameIndex, score] : frames) {
//            std::string fileName = frameIndexToFileName.count(frameIndex)
//                ? frameIndexToFileName.at(frameIndex)
//                : "unknown";
//
//            std::cout << "  File: " << fileName
//                << " (Frame " << frameIndex << ")"
//                << " | HF Score: " << score << std::endl;
//        }
//    }
//
//    // === グリッド描画（高周波エネルギーベース）===
//    for (const auto& [label, frames] : highFreqTopFramesPerLabel) {
//        cv::Mat canvas(thumbRows * thumbHeight, thumbCols * thumbWidth, CV_8UC3, cv::Scalar(255, 255, 255));
//        int count = 0;
//
//        for (const auto& [frameIndex, energy] : frames) {
//            if (!frameIndexToFileName.count(frameIndex)) continue;
//            std::string fileName = frameIndexToFileName[frameIndex];
//            std::string imgPath = VIDEO_FOLDER_PATH + "/" + fileName;
//
//            cv::Mat img = cv::imread(imgPath);
//            if (img.empty()) continue;
//            cv::resize(img, img, cv::Size(thumbWidth, thumbHeight));
//
//            int row = count / thumbCols, col = count % thumbCols;
//            cv::Rect roi(col * thumbWidth, row * thumbHeight, thumbWidth, thumbHeight);
//            img.copyTo(canvas(roi));
//
//            count++;
//            if (count >= TOP_K) break;
//        }
//
//        std::string outPath = THUMBNAIL_PATH + "_HF_thumbnail_grid_label" + std::to_string(label) + ".png";
//        cv::imwrite(outPath, canvas);
//        std::cout << "保存完了（高周波）: " << outPath << std::endl;
//    }
//
//    for (const auto& [label, frames] : topFramesPerLabel) {
//        logStream << "Label " << label << "：\n";
//        for (const auto& f : frames) {
//            std::string fileName = frameIndexToFileName.count(f.frameIndex)
//                ? frameIndexToFileName.at(f.frameIndex) : "unknown";
//            logStream << "  File: " << fileName << " (Frame " << f.frameIndex << ")"
//                << " | S_target: " << f.S_target
//                << " | S_event: " << f.S_event
//                << " | Score: " << (f.S_target - f.S_event) << "\n";
//        }
//    }
//
//    logStream << "\n--- 高周波エネルギーベースのサムネイル候補（上位 " << TOP_K << " 件） ---\n";
//    for (const auto& [label, frames] : highFreqTopFramesPerLabel) {
//        logStream << "Label " << label << "：\n";
//        for (const auto& [frameIndex, score] : frames) {
//            std::string fileName = frameIndexToFileName.count(frameIndex)
//                ? frameIndexToFileName.at(frameIndex) : "unknown";
//            logStream << "  File: " << fileName << " (Frame " << frameIndex << ")"
//                << " | HF Score: " << score << "\n";
//        }
//    }
//
//    saveLogToFile(LOG_FILE_PATH, logStream);
//
//
//    return 0;
//}
