#include <iostream>
#include "config.h"
#include "video_loader.h"
#include "predictor.h"
#include "result_writer.h"

int main() {
    std::vector<cv::Mat> frames;
    if (!loadFramesFromVideo(VIDEO_PATH, frames, FRAME_INTERVAL)) {
        std::cerr << "����̓ǂݍ��݂Ɏ��s���܂����B" << std::endl;
        return -1;
    }

    torch::jit::script::Module model;
    if (!loadModel(TREATMENT_MODEL_PATH, model)) {
        std::cerr << "���f���̓ǂݍ��݂Ɏ��s���܂����B" << std::endl;
        return -1;
    }

    std::vector<std::vector<float>> allProbs;
    for (const auto& frame : frames) {
        allProbs.push_back(predictFrame(frame, model, INPUT_WIDTH, INPUT_HEIGHT));
    }

    if (!saveProbabilitiesToCSV(OUTPUT_CSV, allProbs)) {
        std::cerr << "CSV�t�@�C���̕ۑ��Ɏ��s���܂����B" << std::endl;
        return -1;
    }

    //std::cout << "���������B���ʂ� " << OUTPUT_CSV << " �ɕۑ����܂����B" << std::endl;
    return 0;
}
