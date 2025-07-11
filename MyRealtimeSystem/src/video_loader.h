#pragma once
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>
#include <string>


/*
* @file video_loader.h
* @brief �����摜�t�H���_����t���[����ǂݍ��݁A�O�������s���֐��Q
* @details
* ���̃w�b�_�[�t�@�C���ɂ́A����t�@�C������t���[�����擾����֐���A
* �t�H���_����A�ԉ摜��ǂݍ��ފ֐��A�t���[���̑O�������s���֐����܂܂�Ă��܂��B
* �O�����ł́A�N���b�v��}�X�N�̓K�p�A���T�C�Y�A�J���[�`�����l���ϊ��Ȃǂ��s���A
* �ŏI�I��PyTorch��Tensor�`���ɕϊ����܂��B
*/


// ����t�@�C������t���[�����擾����֐�
bool loadFramesFromVideo(const std::string& videoPath, std::vector<cv::Mat>& frames, int frameInterval);

// �t�H���_����A�ԉ摜��ǂݍ��ށi��Fframe_000.png, ...�j
bool loadFramesFromDirectory(const std::string& folderPath, std::vector<cv::Mat>& frames);

// �t���[����O��������֐�
// �N���b�v�E�}�X�N�̓I�v�V�����Ŏw��\
torch::Tensor preprocessFrame(
	const cv::Mat& frame, 
	int inputWidth, int inputHeight,      // ���̓T�C�Y
	const cv::Rect& cropBox = cv::Rect(), // �N���b�v�{�b�N�X�i�f�t�H���g�͑S�́j
	const cv::Mat& mask = cv::Mat()       // �}�X�N�摜�i�f�t�H���g�͂Ȃ��j
);

// �ǂݍ��񂾉摜��\������(�f�o�b�N�p)
void showFrames(const std::vector<cv::Mat>& frames);