#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 動画ファイルからフレームを取得する関数
bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval);