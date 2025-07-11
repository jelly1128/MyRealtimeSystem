#include "predictor.h"
#include <torch/torch.h>


// ���f����ǂݍ��ފ֐�
bool loadModel(const std::string& modelPath, torch::jit::script::Module& model) {
    try {
        model = torch::jit::load(modelPath);
        model.to(torch::kCUDA);
        model.eval();
        return true;
    }
    catch (const c10::Error& e) {
        return false;
    }
}

// ���u���o���f���̐��_�����s����֐�
std::vector<float> runTreatmentInference(
    const torch::Tensor& frameTensor,
    torch::jit::script::Module& treatmentModel
) {
    torch::NoGradGuard no_grad;                                    // ���z�v�Z�𖳌���
    torch::Tensor input = frameTensor.to(torch::kCUDA);            // GPU�]��
    auto output = treatmentModel.forward({ input }).toTensor();    // ���f���ɓ��͂�n��
	auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU); // �V�O���C�h�֐���K�p���ACPU�ɖ߂�

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}
