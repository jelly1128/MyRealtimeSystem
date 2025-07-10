#include "predictor.h"
#include <torch/torch.h>
#include <iomanip>  // std::setprecision, std::fixed, std::scientific�p

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

static bool saved = false;  // �f�o�b�O�p�t���O

std::vector<float> predictFrame(const cv::Mat& frame, torch::jit::script::Module& model, int inputWidth, int inputHeight) {
    // --- (0) �}�X�N�K�p�i�O�����j ---
    //cv::Mat masked;
    //cv::Mat mask = cv::imread("images/mask.png", cv::IMREAD_GRAYSCALE);
    //if (mask.size() != frame.size()) {
    //    std::cerr << "�}�X�N�T�C�Y�����͉摜�ƈ�v���܂���" << std::endl;
    //    // return ... or resize(mask, ...)
    //}
    //cv::bitwise_and(frame, frame, masked, mask);

    // --- (1) �N���b�v ---
    //cv::Rect cropBox(330, 25, 1260, 970);
    //cv::Mat cropped = masked(cropBox).clone();

    //// --- (2) ���T�C�Y + �J���[�`�����l���ϊ� ---
    //cv::Mat resized, rgb;
    //cv::resize(cropped, resized, cv::Size(inputWidth, inputHeight));  // e.g. 224x224

    // ���T�C�Y�܂łł��Ă���ꍇ
	cv::Mat resized, rgb;
    resized = frame.clone();

    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    if (!rgb.isContinuous()) {
        rgb = rgb.clone();  // from_blob�̑O�ɘA����
    }

    // --- (3) Tensor�ϊ� ---
    torch::Tensor inputTensor = torch::from_blob(
        rgb.data, { 1, inputHeight, inputWidth, 3 }, torch::kFloat32);
    inputTensor = inputTensor.permute({ 0, 3, 1, 2 }).clone();  // NHWC �� NCHW

	// �f�o�b�O�p�ɉ摜��ۑ�
    /*if (!saved) {
		cv::imwrite("debug_frame.png", frame);
		//cv::imwrite("debug_masked.png", masked);
		cv::imwrite("debug_rgb.png", rgb);
        //cv::imwrite("debug_cropped.png", cropped);
        cv::imwrite("debug_resized.png", resized);
        saved = true;
    }*/

    inputTensor = inputTensor.to(torch::kCUDA);

    // --- (5) ���_ ---
	torch::NoGradGuard no_grad;                                    // ���z�v�Z�𖳌���
	auto output = model.forward({ inputTensor }).toTensor();       // ���f���ɓ��͂�n��
	auto probs = torch::sigmoid(output).squeeze().to(torch::kCPU); // �V�O���C�h�֐���K�p���ACPU�ɖ߂�

    std::vector<float> result(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
    return result;
}