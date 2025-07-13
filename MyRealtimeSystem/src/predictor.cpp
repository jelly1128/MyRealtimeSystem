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


// ���핪�ރ��f���̐��_�����s����֐�
int runOrganInference(
    const torch::Tensor& frameTensor, 
    torch::jit::script::Module& organModel,
    torch::Tensor& h_0, torch::Tensor& c_0
) {
    torch::NoGradGuard no_grad;

    // [1, 3, H, W] �� [1, 1, 3, H, W]
    torch::Tensor input = frameTensor.unsqueeze(0);

    // ���f����forward
    auto outputs = organModel.forward({ input.to(torch::kCUDA), h_0, c_0}).toTuple();
    torch::Tensor output = outputs->elements()[0].toTensor()[0]; // ���_�o��
    h_0 = outputs->elements()[1].toTensor(); // �V�����B����
    c_0 = outputs->elements()[2].toTensor(); // �V�����Z�����

    // �ő�v�f�̃C���f�b�N�X�i�������x���j���擾
    int label = output.argmax(1)[0].item<int>(); // shape: [1], [0]��int��

    return label;
}