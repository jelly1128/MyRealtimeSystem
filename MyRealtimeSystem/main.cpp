#include "config.h"
#include "video_loader.h"
#include "predictor.h"
#include "binarizer.h"
#include "result_writer.h"
#include "sliding_window.h"
#include "timeline_writer.h"
#include "thumbnail.h"
#include "thumbnail_selector.h"

#include "debug.h"

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
    }

	timerLoad.stop();*/

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
	std::vector<cv::Mat> processedFramesForTreatment;
	for (cv::Mat& frame : frames) {
		cv::Mat processedFrame = preprocessFrameForTreatment(frame, INPUT_WIDTH, INPUT_HEIGHT);
		processedFramesForTreatment.push_back(processedFrame);
	}

    //showFrames(processedFramesForTreatment, true);  // フレームを表示する関数を呼び出す（デバッグ）

	/*std::vector<cv::Mat> processedFramesForOrgan;
	for (cv::Mat& frame : frames) {
		cv::Mat processedFrame = preprocessFrameForOrgan(frame, INPUT_WIDTH, INPUT_HEIGHT);
		processedFramesForOrgan.push_back(processedFrame);
	}*/

	//timerPreprocess.stop();

	//TimeLogger timerTreatmentInference("処置検出の推論");

	// 3. 推論
	// 処置検出の推論の実行
    /*std::vector<std::vector<float>> treatmentProbabilities;
    for (const cv::Mat& processedFrame : processedFramesForTreatment) {
		treatmentProbabilities.push_back(runTreatmentInference(processedFrame, treatmentModel));
	}*/

	//timerTreatmentInference.stop();

	//TimeLogger timerOrganInference("臓器分類の推論");

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
	//if (!saveMatrixToCSV(TREATMENT_OUTPUT_PROBS_CSV, treatmentProbabilities, "prob_")) {
	//	log("確率CSVの保存に失敗しました。", true);
	//	closeLog();
	//	return -1;
	//} else {
	//	log("推論確率を " + TREATMENT_OUTPUT_PROBS_CSV + " に保存しました。", true);
	//}

	//if (!saveLabelsToCSV(ORGAN_OUTPUT_LABELS_CSV, organLabels)) {
	//	log("臓器分類のラベルCSVの保存に失敗しました。", true);
	//	closeLog();
	//	return -1;
	//}
	//else {
	//	log("臓器分類の推論ラベルを " + ORGAN_OUTPUT_LABELS_CSV + " に保存しました。", true);
	//}

	//timerOrganInference.stop();
	

	// for debug
    std::vector<std::vector<float>> treatmentProbabilities;
	treatmentProbabilities = loadTreatmentProbabilitiesFromCSV(TREATMENT_OUTPUT_PROBS_CSV);
	//std::vector<int> organLabels;
	//organLabels = loadSingleLabelsFromCSV(ORGAN_OUTPUT_LABELS_CSV);

	log("読み込んだ処置確率のサイズ: " + std::to_string(treatmentProbabilities.size()));

	//TimeLogger timerBinarize("バイナリ化");

    // 4. 処理系
	// 推論結果のバイナリ化
	//std::vector<std::vector<int>> treatmentLabels;
	//for (const auto& probs : treatmentProbabilities) {
	//	std::vector<int> binaryLabels = binarizeProbabilities(probs, BINARY_THRESHOLD);
	//	treatmentLabels.push_back(binaryLabels);
	//}

	//timerBinarize.stop();

	//if (!saveMatrixToCSV(TREATMENT_OUTPUT_LABELS_CSV, treatmentLabels, "label_")) {
	//	log("バイナリ化された処置ラベルの保存に失敗しました。", true);
	//	closeLog();
	//	return -1;
	//} else {
	//	log("バイナリ化された処置ラベルを " + TREATMENT_OUTPUT_LABELS_CSV + " に保存しました。", true);
	//}

	//TimeLogger timerSW("スライディングウィンドウによるシーンラベル抽出");

	// スライディングウィンドウを使用してシーンラベルを抽出
	//std::vector<int> treatmentSingleSceneLabels = slidingWindowExtractSceneLabels(treatmentLabels, TREATMENT_SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP, NUM_SCENE_CLASSES);

	//if (!saveLabelsToCSV(TREATMENT_OUTPUT_SCENE_LABELS_CSV, treatmentSingleSceneLabels)) {
	//	log("スライディングウィンドウ適用後の処置ラベルの保存に失敗しました。", true);
	//	closeLog();
	//	return -1;
	//}
	//else {
	//	log("スライディングウィンドウ適用後の処置ラベルを " + TREATMENT_OUTPUT_SCENE_LABELS_CSV + " に保存しました。", true);
	//}

	//timerSW.stop();

	// for debug
	/*std::vector<int> treatmentSingleSceneLabels;
	treatmentSingleSceneLabels = loadSingleLabelsFromCSV(TREATMENT_OUTPUT_SCENE_LABELS_CSV);*/

	//std::cout << "シングルラベルのサイズ: " << treatmentSingleSceneLabels.size() << std::endl;

	// タイムライン画像出力
  //  if (!drawTimelineImage(treatmentSingleSceneLabels, 
		//TREATMENT_TIMELINE_IMAGE_PATH, 
		//NUM_SCENE_CLASSES,
		//TIMELINE_IMAGE_WIDTH,
		//TIMELINE_IMAGE_HEIGHT)
		//) {
  //      std::cerr << "タイムライン画像の出力に失敗しました。" << std::endl;
  //  }
  //  else {
  //      std::cout << "タイムライン画像を " << TREATMENT_TIMELINE_IMAGE_PATH << " に保存しました。" << std::endl;
  //  }

	
	// for debug
	std::vector<int> treatmentSingleSceneLabels;

	// ================== クラス初期化 ==================
	SceneLabelSmoother sceneSmoother(NUM_SCENE_CLASSES, TREATMENT_SLIDING_WINDOW_SIZE);
	SceneSegmentManager segmentManager(THUMNAIL_TOP_K, THUMNAIL_FRAME_GAP);
	FrameWindow frameWindow(TREATMENT_SLIDING_WINDOW_SIZE);

	// ================== 逐次処理 ==================
	for (int i = 0; i < processedFramesForTreatment.size(); ++i) {
		
		// 画像の読み込み（demo）
		cv::Mat processedFrame = processedFramesForTreatment[i];
		// 推論（demo）
		std::vector<float> treatmentProb = treatmentProbabilities[i];

		// バイナリ化とシーンラベルの抽出
		std::vector<int> treatmentBinaryLabels = binarizeProbabilities(treatmentProb, BINARY_THRESHOLD);
		std::vector<int> sceneBinaryLabel(treatmentBinaryLabels.begin(), treatmentBinaryLabels.begin() + NUM_SCENE_CLASSES);
		
		// 推論結果をスライディングウィンドウに追加
		FrameData frameData;
		frameData.frameIndex = i;
		frameData.treatmentProbabilities = treatmentProb;
		frameData.sceneBinaryLabels = sceneBinaryLabel;
		frameData.eventProbsSum = std::accumulate(treatmentProb.begin() + NUM_SCENE_CLASSES, treatmentProb.end(), 0.0f);

		frameWindow.push(processedFrame, frameData);

		// スライディングウィンドウでシーンラベルを決定
		std::optional<int> sceneLabelOpt = sceneSmoother.processSceneLabel(frameWindow.getSceneLabels());

		if (!sceneLabelOpt.has_value()) continue;              // ウィンドウが満たされていない場合はスキップ

		int sceneLabel = sceneLabelOpt.value();
		treatmentSingleSceneLabels.push_back(sceneLabel);      // シーンラベルを保存

		int centerIndex = i - sceneSmoother.getWindowOffset();
		frameWindow.setCenterSceneLabel(sceneLabel);           // スライディングウィンドウの中心フレームのシーンラベルを設定

		// サムネイル候補として登録
		segmentManager.update(frameWindow.getCenterData(), frameWindow.getCenterImage());
	}

	if (!saveLabelsToCSV(TREATMENT_OUTPUT_SCENE_LABELS_CSV, treatmentSingleSceneLabels)) {
		log("スライディングウィンドウ適用後の処置ラベルの保存に失敗しました。", true);
		closeLog();
		return -1;
	}
	else {
		log("スライディングウィンドウ適用後の処置ラベルを " + TREATMENT_OUTPUT_SCENE_LABELS_CSV + " に保存しました。", true);
	}

	// サムネイル選定の終了
	segmentManager.finalize();

	// サムネイル候補の出力
	auto thumbs = segmentManager.getFinalThumbnails();

	// 可視化
	visualizeThumbnailsPerLabel(thumbs, TREATMENT_THUMNAIL_IMAGE_PATH, INPUT_WIDTH, INPUT_HEIGHT);

	// log出力
	//segmentManager.logSummary();

	timerAll.stop();
	closeLog();
    return 0;
}