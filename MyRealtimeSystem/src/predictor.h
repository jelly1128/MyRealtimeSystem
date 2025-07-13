#pragma once
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

/**
 * @brief PyTorchモデルを読み込む
 * @param[in] modelPath モデルファイルのパス
 * 
 * @retval true 読み込み成功
 * @retval false 読み込み失敗
 */
 // モデルを読み込む関数
bool loadModel(
    const std::string& modelPath,
    torch::jit::script::Module& model
);


/** 
 * @brief 1フレームのテンソルでモデル推論を実行し、確率ベクトルを返す
 * @param[in] frameTensor 前処理済みの画像テンソル (1,3,H,W)
 * @param[in] model 推論用PyTorchモデル
 * @return 各クラスの確率（float型ベクトル）
 */
 // 処置検出モデルの推論を実行する関数
std::vector<float> runTreatmentInference(
    const torch::Tensor& frameTensor, 
    torch::jit::script::Module& treatmentModel
);

/**
 * @brief 臓器分類LSTMモデルで推論し、確率ベクトルを返す
 * @param[in] frameTensor 前処理済みの画像テンソル (1,3,H,W)
 * @param[in] model 臓器分類PyTorchモデル
 * @param[in,out] h_0 LSTMの隠れ状態
 * @param[in,out] c_0 LSTMのセル状態
 * @return 各クラスの確率（float型ベクトル）
 */
int runOrganInference(
    const torch::Tensor& frameTensor,
    torch::jit::script::Module& model,
    torch::Tensor& h_0, torch::Tensor& c_0
);