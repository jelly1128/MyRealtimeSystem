#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 動画ファイルからフレームを取得する関数
bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval);

// フォルダから連番画像を読み込む（例：frame_000.png, ...）
bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames);