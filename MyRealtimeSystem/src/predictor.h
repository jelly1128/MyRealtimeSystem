#pragma once
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

/**
 * @brief PyTorch���f����ǂݍ���
 * @param[in] modelPath ���f���t�@�C���̃p�X
 * @param[out] model �ǂݍ��񂾃��f��
 * @retval true �ǂݍ��ݐ���
 * @retval false �ǂݍ��ݎ��s
 */
 // ���f����ǂݍ��ފ֐�
bool loadModel(const std::string& modelPath, torch::jit::script::Module& model);

/**
 * @brief 1�t���[���̃e���\���Ń��f�����_�����s���A�m���x�N�g����Ԃ�
 * @param[in] frameTensor �O�����ς݂̉摜�e���\�� (1,3,H,W)
 * @param[in] model ���_�pPyTorch���f��
 * @return �e�N���X�̊m���ifloat�^�x�N�g���j
 */
// ���_�����s����֐�
std::vector<float> runTreatmentInference(const torch::Tensor& frameTensor, torch::jit::script::Module& treatmentModel);
