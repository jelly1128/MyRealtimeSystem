#include "debug.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

// CSV�t�@�C������t���[���̊m����ǂݍ��ފ֐�
//�g����
// for debug
// std::vector<std::vector<float>> frameProbabilities;
// frameProbabilities = loadFrameProbabilitiesFromCSV(DEBUG_PROBS_CSV);
std::vector<std::vector<float>> loadFrameProbabilitiesFromCSV(const std::string& csvPath) {
    std::vector<std::vector<float>> probabilities;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] �t�@�C�����J���܂���: " << csvPath << std::endl;
        return probabilities;  // ��̃x�N�g����Ԃ�
    }
    std::string line;
    // ��s�ځi�w�b�_�[�j���X�L�b�v
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        int colIndex = 0;
        while (std::getline(ss, value, ',')) {
            if (colIndex > 0) {  // 2��ڈȍ~�����ǂݍ���
                try {
                    row.push_back(std::stof(value));
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number: " << value << std::endl;
                    row.push_back(0.0f);  // �G���[�����i�K�v�ɉ����ĕύX�j
                }
            }
            ++colIndex;
        }
        if (!row.empty()) {
            probabilities.push_back(row);
		}
    }
    if (probabilities.empty()) {
        std::cerr << "[DEBUG] �m����1���ǂݍ��܂�܂���ł����B" << std::endl;
    }
	return probabilities;
}


bool loadMainLabelsFromCSV(const std::string& csvPath, std::vector<int>& labels) {
    labels.clear();
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] �t�@�C�����J���܂���: " << csvPath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string filename, labelStr;
        // 1��ڂ��X�L�b�v���A2��ڂ��擾
        if (std::getline(ss, filename, ',') && std::getline(ss, labelStr, ',')) {
            try {
                int labelValue = std::stoi(labelStr);
                labels.push_back(labelValue);
            } catch (...) {
                std::cerr << "[DEBUG] �����Ȓl���X�L�b�v(2���): " << labelStr << std::endl;
                continue;
            }
        }
    }
    if (labels.empty()) {
        std::cerr << "[DEBUG] ���x����1���ǂݍ��܂�܂���ł����B" << std::endl;
    }
    return !labels.empty();
}