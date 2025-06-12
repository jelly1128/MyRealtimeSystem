#pragma once
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

// モデルの初期化とフレーム1枚の推論
bool loadModel(const std::string& modelPath, torch::jit::script::Module& model);
std::vector<float> predictFrame(const cv::Mat& frame, torch::jit::script::Module& model, int inputWidth, int inputHeight);
