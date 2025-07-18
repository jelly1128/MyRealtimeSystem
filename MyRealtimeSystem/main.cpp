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
	std::vector<cv::Mat> processedFramesForTreatment;
	for (cv::Mat& frame : frames) {
		cv::Mat processedFrame = preprocessFrameForTreatment(frame, INPUT_WIDTH, INPUT_HEIGHT);
		processedFramesForTreatment.push_back(processedFrame);
	}

	//std::ofstream ofs(HIGH_FREQ_CSV);
	//if (!ofs.is_open()) {
	//	std::cerr << "Failed to open output CSV file." << std::endl;
	//	return 0;
	//}

	//// ヘッダー
	//ofs << "frameIndex,highFrequencyScore\n";
	//// 処理と保存
	//int frameIndex = 0;
	//for (const cv::Mat& frame : processedFramesForTreatment) {
	//	// 高周波エネルギーの算出
	//	float highFrequencyScore = computeHighFrequencyEnergy(frame);
	//	// CSVに保存
	//	ofs << frameIndex << "," << highFrequencyScore << "\n";

	//	frameIndex++;
	//}

	//ofs.close();

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

	// 動画内の各フレームの推論デモ
	for (int i = 0; i < treatmentProbabilities.size(); ++i) {
		// 入力画像
		//cv::Mat inputImage = frames[i];
		cv::Mat inputImage = processedFramesForTreatment[i];
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
			candidate.deepLearningScore = VideoFrameData[centerIndex].sceneProb - VideoFrameData[centerIndex].eventProbsSum / 2;  // シーンラベルの確率をスコアとして使用
			candidate.highFrequencyScore = computeHighFrequencyEnergy(candidate.frame);  // 高周波エネルギーを計算

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
			// サムネイル画像を表示
			if (!cand.frame.empty()) {
				cv::Mat bgr;
				cv::cvtColor(cand.frame, bgr, cv::COLOR_RGB2BGR);  // OpenCVのBGRをRGBに変換
				cv::imshow("Thumbnail", bgr);
				cv::waitKey(0);  // キー入力待ち
			} else {
				std::cerr << "Empty image at frameIndex: " << cand.frameIndex << std::endl;
			}
		}
		visualizeThumbnailsPerLabel(finalThumbnailsPerLabel, TREATMENT_THUMNAIL_IMAGE_PATH, INPUT_WIDTH, INPUT_HEIGHT);
	}

	log("=== ラベルごとの最長区間 ===", true);
	for (const auto& [label, seg] : longestSegment) {
		log("Label " + std::to_string(label)
			+ " [" + std::to_string(seg.startFrameIndex) + "," + std::to_string(seg.endFrameIndex) + "]"
			+ " (length=" + std::to_string(seg.length) + ")", true);

		// priority_queue から vector に抜き出す
		std::priority_queue<ThumbnailCandidate> thumbs = seg.topKThumbnails;
		std::vector<ThumbnailCandidate> thumbsVec;
		while (!thumbs.empty()) {
			thumbsVec.push_back(thumbs.top());
			thumbs.pop();
		}

		// スコア降順でsort（念のため明示的に！）
		std::sort(thumbsVec.begin(), thumbsVec.end(),
			[](const ThumbnailCandidate& a, const ThumbnailCandidate& b) {
				return a.combinedScore() > b.combinedScore();
			}
		);

		// ソートした上で表示
		for (const auto& cand : thumbsVec) {
			log("Frame " + std::to_string(cand.frameIndex)
				+ ", DeepLearningScore=" + std::to_string(cand.deepLearningScore)
				+ ", HiFreqScore=" + std::to_string(cand.highFrequencyScore)
				+ ", 合成スコア=" + std::to_string(cand.combinedScore()), true);
		}
	}

	// 動画全体のフレーム情報をlogに出力
	/*for (const FrameData& frame : VideoFrameData) {
		log("フレーム " + std::to_string(frame.frameIndex) +
			": シーンラベル = " + std::to_string(frame.sceneLabel) +
			": シーンラベルの確率 = " + std::to_string(frame.sceneProb) +
			": イベントラベルの確率の合計 = " + std::to_string(frame.eventProbsSum), true);
	}*/

	timerAll.stop();
	closeLog();
    return 0;
}