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
    std::regex pattern(R"(_(\d+)\.png$)");
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);  // マッチした番号を整数に変換
    }
    return -1;  // 見つからなかった場合（先頭に来ないように）
}


// フォルダからフレームを読み込む関数
bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames) {
    frames.clear();
    std::vector<cv::String> filepaths;

    cv::glob(folderPath + "/*.png", filepaths, false);
    if (filepaths.empty()) return false;

    // フレーム番号でソート
    std::sort(filepaths.begin(), filepaths.end(), [](const cv::String& a, const cv::String& b) {
        return extractFrameNumber(a) < extractFrameNumber(b);
        });

    for (const auto& path : filepaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            frames.push_back(img);
        }
        else {
            std::cerr << "Failed to load image: " << path << std::endl;
        }
    }

    return !frames.empty();
}


// 画像前処理
torch::Tensor preprocessFrame(
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
    //{
    //    cv::Mat debugImg, debugShow;
    //    rgb.convertTo(debugImg, CV_8U, 255.0); // float → uchar
    //    cv::cvtColor(debugImg, debugShow, cv::COLOR_RGB2BGR); // RGB→BGRに戻す
    //    if (!debugShow.empty()) {
    //        cv::imshow("Debug RGB", debugShow);
    //        int key = cv::waitKey(0);
    //    }
    //    else {
    //        std::cerr << "debugShow is empty! Cannot show window." << std::endl;
    //    }
    //}
    // ==================

	// --- (5) Tensor変換 ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC → NCHW

    inputTensor = inputTensor.to(torch::kCUDA);

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