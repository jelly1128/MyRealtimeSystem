#include "debug.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

bool loadLabelsFromCSV(const std::string& csvPath, std::vector<int>& labels) {
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