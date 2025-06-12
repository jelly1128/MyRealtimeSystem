#include <iostream>
#include "config.h"
#include "src/video_loader.h"
#include "src/predictor.h"
#include "src/result_writer.h"

int main() {
	// ����t�@�C���̃p�X�ƃ��f���̃p�X��ݒ�
    std::vector<cv::Mat> frames;
    if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
        std::cerr << "����̓ǂݍ��݂Ɏ��s���܂����B" << std::endl;
        return -1;
    }

	// ���f���̓ǂݍ���
    torch::jit::script::Module model;
    if (!loadModel(TREATMENT_MODEL_PATH, model)) {
        std::cerr << "���f���̓ǂݍ��݂Ɏ��s���܂����B" << std::endl;
        return -1;
    }

	// ���_�̎��s
    std::vector<std::vector<float>> allProbs;
    for (const auto& frame : frames) {
        allProbs.push_back(predictFrame(frame, model, INPUT_WIDTH, INPUT_HEIGHT));
    }

	// ���ʂ�CSV�t�@�C���ւ̕ۑ�
    if (!saveProbabilitiesToCSV(OUTPUT_CSV, allProbs)) {
        std::cerr << "CSV�t�@�C���̕ۑ��Ɏ��s���܂����B" << std::endl;
        return -1;
    }

    //std::cout << "���������B���ʂ� " << OUTPUT_CSV << " �ɕۑ����܂����B" << std::endl;
    return 0;
}
