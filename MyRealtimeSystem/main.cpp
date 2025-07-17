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
	initLog(LOG_FILE_PATH);
	log("プログラム開始", true);
	TimeLogger timerAll("全体処理時間");

	//TimeLogger timerLoad("モデル読み込み");

	// 1. 初期化
    // モデルの読み込み
    /*torch::jit::script::Module treatmentModel, organModel;
    if (!loadModel(TREATMENT_MODEL_PATH, treatmentModel) ||
		!loadModel(ORGAN_MODEL_PATH, organModel)) {
		log("モデルの読み込みに失敗しました。", true);
		closeLog();
        return -1;
    }*/

	//timerLoad.stop();

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

	//TimeLogger timerRead("フレーム読み込み");

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

	//timerRead.stop();

	//TimeLogger timerPreprocess("フレーム前処理");

	// 画像前処理
	/*std::vector<cv::Mat> processedFramesForTreatment;
	for (cv::Mat& frame : frames) {
		cv::Mat processedFrame = preprocessFrameForTreatment(frame, INPUT_WIDTH, INPUT_HEIGHT);
		processedFramesForTreatment.push_back(processedFrame);
	}*/

    //showFrames(processedFramesForTreatment, true);  // フレームを表示する関数を呼び出す（デバッグ）

	/*std::vector<cv::Mat> processedFramesForOrgan;
	for (cv::Mat& frame : frames) {
		cv::Mat processedFrame = preprocessFrameForOrgan(frame, INPUT_WIDTH, INPUT_HEIGHT);
		processedFramesForOrgan.push_back(processedFrame);
	}*/

	//timerPreprocess.stop();

	//TimeLogger timerInference("処置検出の推論");

	// 3. 推論
	// 処置検出の推論の実行
    /*std::vector<std::vector<float>> treatmentProbabilities;
    for (const cv::Mat& processedFrame : processedFramesForTreatment) {
		treatmentProbabilities.push_back(runTreatmentInference(processedFrame, treatmentModel));
	}*/

	// 隠れ状態とセル状態の初期化
	//torch::Tensor h_0 = torch::zeros({ 2, 1, 128 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
	//torch::Tensor c_0 = torch::zeros({ 2, 1, 128 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

	// 臓器分類の推論の実行
	//std::vector<int> organLabels;
	//for (const cv::Mat& processedFrame : processedFramesForOrgan) {
	//	// 推論部分やfrom_blobの周辺
	//	int label = runOrganInference(processedFrame, organModel, h_0, c_0);
	//	organLabels.push_back(label);
	//}


	// 推論結果の保存
	/*if (!saveMatrixToCSV(TREATMENT_OUTPUT_PROBS_CSV, treatmentProbabilities, "prob_")) {
		log("確率CSVの保存に失敗しました。", true);
		closeLog();
		return -1;
	} else {
		log("推論確率を " + TREATMENT_OUTPUT_PROBS_CSV + " に保存しました。", true);
	}*/

	/*if (!saveLabelsToCSV(ORGAN_OUTPUT_LABELS_CSV, organLabels)) {
		log("臓器分類のラベルCSVの保存に失敗しました。", true);
		closeLog();
		return -1;
	}
	else {
		log("臓器分類の推論ラベルを " + ORGAN_OUTPUT_LABELS_CSV + " に保存しました。", true);
	}*/

	//timerInference.stop();

	// for debug
    std::vector<std::vector<float>> treatmentProbabilities;
	treatmentProbabilities = loadTreatmentProbabilitiesFromCSV(TREATMENT_OUTPUT_PROBS_CSV);
	/*std::vector<int> organLabels;
	organLabels = loadSingleLabelsFromCSV(ORGAN_OUTPUT_LABELS_CSV);*/

	//std::cout << treatmentProbabilities.size() << " frames loaded." << std::endl;

	//TimeLogger timerBinarize("バイナリ化");

    // 4. 処理系
	// 推論結果のバイナリ化
	/*std::vector<std::vector<int>> treatmentLabels;
	for (const auto& probs : treatmentProbabilities) {
		std::vector<int> binaryLabels = binarizeProbabilities(probs, BINARY_THRESHOLD);
		treatmentLabels.push_back(binaryLabels);
	}*/

	//timerBinarize.stop();

	/*if (!saveMatrixToCSV(TREATMENT_OUTPUT_LABELS_CSV, treatmentLabels, "label_")) {
		log("バイナリ化された処置ラベルの保存に失敗しました。", true);
		closeLog();
		return -1;
	} else {
		log("バイナリ化された処置ラベルを " + TREATMENT_OUTPUT_LABELS_CSV + " に保存しました。", true);
	}*/

	//TimeLogger timerSW("スライディングウィンドウによるシーンラベル抽出");

	// スライディングウィンドウを使用してシーンラベルを抽出
	/*std::vector<int> treatmentSingleSceneLabels = slidingWindowExtractSceneLabels(treatmentLabels, TREATMENT_SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, NUM_SCENE_CLASSES);

	if (!saveLabelsToCSV(TREATMENT_OUTPUT_SCENE_LABELS_CSV, treatmentSingleSceneLabels)) {
		log("スライディングウィンドウ適用後の処置ラベルの保存に失敗しました。", true);
		closeLog();
		return -1;
	}
	else {
		log("スライディングウィンドウ適用後の処置ラベルを " + TREATMENT_OUTPUT_SCENE_LABELS_CSV + " に保存しました。", true);
	}*/

	//timerSW.stop();

	// for debug
	//std::vector<int> treatmentSingleSceneLabels;
	//treatmentSingleSceneLabels = loadSingleLabelsFromCSV(TREATMENT_OUTPUT_SCENE_LABELS_CSV);

	//std::cout << "シングルラベルのサイズ: " << treatmentSingleSceneLabels.size() << std::endl;

	// タイムライン画像出力
    /*if (!drawTimelineImage(treatmentSingleSceneLabels, 
		TREATMENT_TIMELINE_IMAGE_PATH, 
		NUM_SCENE_CLASSES,
		TIMELINE_IMAGE_WIDTH,
		TIMELINE_IMAGE_HEIGHT)
		) {
        std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
    }
    else {
        std::cout << "タイムライン画像を " << TREATMENT_TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
    }*/

	// for debug
	// サムネイル選定の実施
	// 動画全体のフレーム情報
	std::vector<FrameData> VideoFrameData;
	std::map<int, std::priority_queue<ThumbnailCandidate>> topKThumbnails;

	// 今連続中の区間
	std::map<int, VideoSegment> currentSegment;

	// これまでで最長だった区間
	std::map<int, VideoSegment> longestSegment;

	// スライディングウィンドウ用の状態（履歴と前回ラベル）
	std::deque<std::vector<int>> windowSceneLabelBuffer;
	int prevSceneLabel = -1;
	const int halfWindowSize = TREATMENT_SLIDING_WINDOW_SIZE / 2; // ウィンドウの半分のサイズ
	std::unordered_map<int, cv::Mat> windowFrameBuffer; // (frameIndex, image)のペア
	std::deque<int> windowIndices;

	std::cout << halfWindowSize << " frames for sliding window." << std::endl;

	// 動画内の各フレームの推論デモ
	for (int i = 0; i < treatmentProbabilities.size(); ++i) {
		// 入力画像
		cv::Mat inputImage = frames[i];
		// 処置確率
		std::vector<float> treatmentProb = treatmentProbabilities[i];

		// 情報更新
		FrameData frameData;
		frameData.frameIndex = i;
		frameData.treatmentProbabilities = treatmentProb;
		frameData.eventProbsSum = std::accumulate(treatmentProb.begin() + NUM_SCENE_CLASSES, treatmentProb.end(), 0.0f);

		VideoFrameData.push_back(frameData);

		// バイナリシーンラベルを抽出
		std::vector<int> treatmentBinaryLabels = binarizeProbabilities(treatmentProb, BINARY_THRESHOLD);
		std::vector<int> sceneBinaryLabel(treatmentBinaryLabels.begin(), treatmentBinaryLabels.begin() + NUM_SCENE_CLASSES);

		// スライディングウィンドウバッファ更新
		windowSceneLabelBuffer.push_back(sceneBinaryLabel);
		windowFrameBuffer[i] = inputImage.clone();
		windowIndices.push_back(i);

		std::cout << "frame " << std::to_string(i) << " : " << windowFrameBuffer.size() << " frames in window buffer. " << std::to_string(windowIndices.front()) << ": frame size" << std::endl;

		if (windowIndices.size() > TREATMENT_SLIDING_WINDOW_SIZE) {
			int oldestIndex = windowIndices.front();
			windowIndices.pop_front();
			windowFrameBuffer.erase(oldestIndex); // 古いフレームを削除
		}
		
		if (windowSceneLabelBuffer.size() > TREATMENT_SLIDING_WINDOW_SIZE) {
			windowSceneLabelBuffer.pop_front();
		} else if (windowSceneLabelBuffer.size() < TREATMENT_SLIDING_WINDOW_SIZE) {
			// ウィンドウサイズに満たない場合
			continue;
		}



		// スライディングウィンドウの適用により、中心フレームのシーンラベルを決定
		auto windowCenterLabel = processSceneLabelSlidingWindow(windowSceneLabelBuffer, prevSceneLabel);

		if (windowCenterLabel.has_value()) {
			int centerIndex = i - halfWindowSize;           // ウィンドウの中心フレームのインデックス
			int centerLabel = windowCenterLabel.value();    // スライディングウィンドウで決定したラベル

			// スライディングウィンドウの中心フレームのシーンラベルを設定
			VideoFrameData[centerIndex].sceneLabel = centerLabel;
			// シーンラベルの確率を設定
			VideoFrameData[centerIndex].sceneProb = VideoFrameData[centerIndex].treatmentProbabilities[centerLabel];
			//prevSceneLabel = centerLabel;

			// --- 動画セグメントの更新・サムネイル管理の追加 ----
			ThumbnailCandidate candidate;
			candidate.frameIndex = centerIndex;
			candidate.frame = windowFrameBuffer.find(centerIndex)->second.clone();  // ウィンドウ内のフレーム画像
			candidate.deepLearningScore = VideoFrameData[centerIndex].sceneProb - VideoFrameData[centerIndex].eventProbsSum;  // シーンラベルの確率をスコアとして使用
			candidate.highFrequencyScore = 0.0f;  // 高周波エネルギーのスコアは後で計算

			// 区間管理
			if (prevSceneLabel == -1 || centerLabel != prevSceneLabel) {
				// 直前のラベルと違う場合，直前のラベル区間を最長比較・更新
				if (prevSceneLabel != -1 && currentSegment.count(prevSceneLabel)) {
					if (currentSegment[prevSceneLabel].length > longestSegment[prevSceneLabel].length) {
						longestSegment[prevSceneLabel] = currentSegment[prevSceneLabel];
					}
					currentSegment.clear();  // 現在のセグメントをクリア
				}
				// 新しいラベル区間を開始
				VideoSegment newSegment;
				newSegment.startFrameIndex = centerIndex;
				newSegment.endFrameIndex = centerIndex;
				newSegment.length = 1;  // 初期は1フレーム
				newSegment.topKThumbnails.push(candidate);  // 初期のサムネイル候補を追加
				currentSegment[centerLabel] = newSegment;
			} else {
				// 直前のラベルと同じ場合、区間を更新
				currentSegment[centerLabel].endFrameIndex = centerIndex;
				currentSegment[centerLabel].length++;
				currentSegment[centerLabel].topKThumbnails.push(candidate);
				// サムネイル候補がTOP_Kを超えた場合、最小値を削除
				if (currentSegment[centerLabel].topKThumbnails.size() > THUMNAIL_TOP_K) {
					currentSegment[centerLabel].topKThumbnails.pop();
				}
			}
			prevSceneLabel = centerLabel;
		}
	}

	// ループ後、最後の区間も最長更新を忘れずに
	for (auto& [label, seg] : currentSegment) {
		if (seg.length > longestSegment[label].length) {
			longestSegment[label] = seg;
		}
	}

	std::map<int, std::vector<ThumbnailCandidate>> finalThumbnailsPerLabel;

	for (const auto& [label, seg] : longestSegment) {
		// seg.topKThumbnailsは区間内スコア順priority_queue
		auto selected = selectThumbnailsWithFrameGap(seg.topKThumbnails, THUMNAIL_FRAME_GAP, THUMNAIL_TOP_K);
		finalThumbnailsPerLabel[label] = selected;

		// デバッグ用出力
		log("Label " + std::to_string(label) + " サムネイル選定:", true);
		for (const auto& cand : selected) {
			log(" " + std::to_string(cand.frameIndex), true);
		}

		// topKThumbnailsなどのサムネイル候補を表示したい場合
		auto thumbnails = seg.topKThumbnails;
		while (!thumbnails.empty()) {
			const ThumbnailCandidate& candidate = thumbnails.top();
			if (candidate.frame.empty()) {
				std::cerr << "Empty image at frameIndex: " << candidate.frameIndex << std::endl;
				thumbnails.pop();
				continue;
			}
			cv::imshow("Thumbnail", candidate.frame);
			cv::waitKey(0);
			thumbnails.pop();
		}
	}

	log("=== ラベルごとの最長区間 ===", true);
	for (const auto& [label, seg] : longestSegment) {
		log("Label " + std::to_string(label)
			+ " [" + std::to_string(seg.startFrameIndex) + "," + std::to_string(seg.endFrameIndex) + "]"
			+ " (length=" + std::to_string(seg.length) + ")", true);

		std::priority_queue<ThumbnailCandidate> thumbs = seg.topKThumbnails;
		std::vector<float> scores;
		while (!thumbs.empty()) {
			const auto& cand = thumbs.top();
			log("Frame " + std::to_string(cand.frameIndex)
				+ ", DeepLearningScore=" + std::to_string(cand.deepLearningScore)
				+ ", HiFreqScore=" + std::to_string(cand.highFrequencyScore)
				+ ", 合成スコア=" + std::to_string(cand.combinedScore()), true);
			scores.push_back(cand.combinedScore());
			thumbs.pop();
		}
		// ソートの確認
		for (size_t i = 1; i < scores.size(); ++i) {
			if (scores[i] > scores[i - 1]) {
				std::cout << "  ※警告：スコアが降順になっていません！" << std::endl;
			}
		}
	}

	// 動画全体のフレーム情報をlogに出力
	for (const FrameData& frame : VideoFrameData) {
		log("フレーム " + std::to_string(frame.frameIndex) +
			": シーンラベル = " + std::to_string(frame.sceneLabel) +
			": シーンラベルの確率 = " + std::to_string(frame.sceneProb) +
			": イベントラベルの確率の合計 = " + std::to_string(frame.eventProbsSum), true);
	}

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
