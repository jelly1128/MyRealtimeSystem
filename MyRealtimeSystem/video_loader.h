#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// ����t�@�C������t���[�����擾����֐�
bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval);