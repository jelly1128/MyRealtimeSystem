#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// ����t�@�C������t���[�����擾����֐�
bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval);

// �t�H���_����A�ԉ摜��ǂݍ��ށi��Fframe_000.png, ...�j
bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames);

// �ǂݍ��񂾉摜��\������(�f�o�b�N�p)
void showFrames(const std::vector<cv::Mat>& frames);