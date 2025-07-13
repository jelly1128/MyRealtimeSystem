#include "video_loader.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <regex>


// 動画からフレームを読み込む関数
bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        return false;
    }

    int frameCount = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frameCount % frameInterval == 0) {
            frames.push_back(frame.clone());  // 安全にコピー
        }
        frameCount++;
    }

    cap.release();
    return true;
}


// フレーム番号を抽出する関数
int extractFrameNumber(const std::string& filename) {
    std::smatch match;
    // _数字.png または 数字.png のどちらにもマッチ
    std::regex pattern(R"((?:_|/)?(\d+)\.png$)");
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);  // マッチした番号を整数に変換
    }
    return -1;  // 見つからなかった場合（先頭に来ないように）
}


// フォルダからフレームを読み込む関数
bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
    frames.clear();
    std::vector<cv::String> filepaths;

	//std::vector<int> frameNumbers; // for debug: フレーム番号を保持するためのベクター

    cv::glob(folderPath + "/*.png", filepaths, false);
    if (filepaths.empty()) return false;

    // フレーム番号でソート
    std::sort(filepaths.begin(), filepaths.end(), [](const cv::String& a, const cv::String& b) {
        return extractFrameNumber(a) < extractFrameNumber(b);
        });

    for (const auto& path : filepaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        int frameNum = extractFrameNumber(path);
        if (!img.empty()) {
            frames.push_back(img);
			//frameNumbers.push_back(frameNum); // for debug: フレーム番号を追加
        }
        else {
            std::cerr << "Failed to load image: " << path << std::endl;
        }
    }

	// for debug: フレーム番号を表示
    /*std::cout << "Loaded frame numbers: ";
    for (auto num : frameNumbers) std::cout << num << " ";
    std::cout << std::endl;*/

    return !frames.empty();
}


// 画像前処理
torch::Tensor preprocessFrameForTreatment(
    const cv::Mat& frame,
    int inputWidth, int inputHeight, // リサイズ後のサイズ
    const cv::Rect& cropBox,         // クロップ領域（未指定なら全体)
    const cv::Mat& mask              // マスク画像（空なら未使用）
) {
    frame.clone();

    // --- (1) マスク適用 ---
    cv::Mat masked;
    if (!mask.empty() && mask.size() == frame.size()) {
        cv::bitwise_and(frame, frame, masked, mask);
    } else {
        masked = frame.clone();
    }

	// --- (2) クロップ ---
    cv::Mat cropped;
    if (cropBox.width > 0 && cropBox.height > 0 &&
        cropBox.x >= 0 && cropBox.y >= 0 &&
        cropBox.x + cropBox.width <= masked.cols &&
        cropBox.y + cropBox.height <= masked.rows) {
        cropped = masked(cropBox).clone();
	} else {
        cropped = masked.clone();  // クロップしない場合は全体
    }

	// --- (3) リサイズ ---
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));
	resized = resized.clone();  // リサイズ後の画像を確保

	// --- (4) カラーチャンネル変換 ---
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
	rgb = rgb.clone();  // 連続化のためにclone

    // === デバッグ表示 ===
    /*cv::imshow("1. Masked", masked);
    std::cout << "Masked size: " << masked.cols << "x" << masked.rows << std::endl;

    cv::imshow("2. Crop", cropped);
    std::cout << "Cropped size: " << cropped.cols << "x" << cropped.rows << std::endl;

    cv::imshow("3. Resized", resized);
    std::cout << "Resized size: " << resized.cols << "x" << resized.rows << std::endl;

    cv::waitKey(0);*/
    // ==================

	// --- (5) Tensor変換 ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC → NCHW

    inputTensor = inputTensor.to(torch::kCUDA);

    return inputTensor;
}


// 画像前処理
torch::Tensor preprocessFrameForOrgan(
    const cv::Mat& frame,
    int inputWidth, int inputHeight, // リサイズ後のサイズ
	const int targetShort,           // 短辺の目標サイズ
    const cv::Rect& cropBox,         // クロップボックス（デフォルトは全体）
    const cv::Mat& mask              // マスク画像（空なら未使用）
) {
    frame.clone();

    // --- (1) マスク適用（frame全体にmask。maskサイズはframeと同じ） ---
    cv::Mat masked;
    if (!mask.empty() && mask.size() == frame.size()) {
        cv::bitwise_and(frame, frame, masked, mask);
    }
    else {
        masked = frame.clone();
    }

    // --- (2) cropBoxでクロップ（ここで部分抽出） ---
    cv::Mat croppedFirst;
    if (cropBox.width > 0 && cropBox.height > 0 &&
        cropBox.x >= 0 && cropBox.y >= 0 &&
        cropBox.x + cropBox.width <= masked.cols &&
        cropBox.y + cropBox.height <= masked.rows) {
        croppedFirst = masked(cropBox).clone();
    }
    else {
        croppedFirst = masked.clone();  // クロップしない場合は全体
    }

    // --- (3) SmallestMaxSize（短辺=targetShortでアスペクト比維持リサイズ） ---
    int origW = croppedFirst.cols;
    int origH = croppedFirst.rows;
    float scale = (float)targetShort / std::min(origW, origH);
    int resizedW = std::round(origW * scale);
    int resizedH = std::round(origH * scale);

    cv::Mat resized;
    cv::resize(croppedFirst, resized, cv::Size(resizedW, resizedH));

    // --- (4) 中央inputWidth×inputHeightクロップ ---
    int x = (resized.cols - inputWidth) / 2; // 338-224=114 → x=57
    int y = (resized.rows - inputHeight) / 2; // 270-224=46 → y=23

    cv::Rect centerCropBox(x, y, inputWidth, inputHeight);
    cv::Mat cropped = resized(centerCropBox).clone();

    // --- (5) カラーチャンネル変換 ---
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    rgb = rgb.clone();  // 連続化のためにclone

    // === デバッグ表示 ===
    // デバッグウィンドウをすべて表示
    /*cv::imshow("1. Masked", masked);
	std::cout << "Masked size: " << masked.cols << "x" << masked.rows << std::endl;

    cv::imshow("2. Cropped First", croppedFirst);
    std::cout << "Cropped size: " << croppedFirst.cols << "x" << croppedFirst.rows << std::endl;

    cv::imshow("3. Resized", resized);
    std::cout << "Resized size: " << resized.cols << "x" << resized.rows << std::endl;

    cv::imshow("4. Center Cropped", cropped);
    std::cout << "Cropped size after center crop: " << cropped.cols << "x" << cropped.rows << std::endl;

    cv::waitKey(0);*/
    // ==================

    // --- (6) Tensor変換 ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC → NCHW

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (7) 正規化 ---
    torch::Tensor mean = torch::tensor({ 0.5, 0.5, 0.5 }).view({ 1, 3, 1, 1 }).to(torch::kCUDA);
    torch::Tensor std = torch::tensor({ 0.5, 0.5, 0.5 }).view({ 1, 3, 1, 1 }).to(torch::kCUDA);
    inputTensor = (inputTensor - mean) / std;

    return inputTensor;
}


// 読み込んだ画像を表示する(デバック用)
void showFrames(const std::vector<cv::Mat>& frames) {
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::imshow("Frame", frames[i]);
        int key = cv::waitKey(500); // 500msごとに次の画像へ
        if (key == 27) break; // ESCキーで中断
    }
    cv::destroyAllWindows();
}