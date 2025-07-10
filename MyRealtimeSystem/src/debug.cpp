#include "debug.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>

namespace {
    std::ofstream g_logFile;
    std::mutex g_logMutex;
}

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
            try {
                row.push_back(std::stof(value));
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number: " << value << std::endl;
                row.push_back(-1.0);  // �G���[�����i�K�v�ɉ����ĕύX�j
            }
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


std::vector<std::vector<int>> loadFrameBinariesFromCSV(const std::string& csvPath) {
    std::vector<std::vector<int>> binaries;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] �t�@�C�����J���܂���: " << csvPath << std::endl;
        return binaries;  // ��̃x�N�g����Ԃ�
    }
    std::string line;
    // ��s�ځi�w�b�_�[�j���X�L�b�v
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<int> row;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stoi(value));
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number: " << value << std::endl;
                row.push_back(-1);
            }
        }
        if (!row.empty()) {
            binaries.push_back(row);
        }
    }
    if (binaries.empty()) {
        std::cerr << "[DEBUG] �o�C�i����1���ǂݍ��܂�܂���ł����B" << std::endl;
    }
	return binaries;
}


std::vector<int> loadWindowedSceneLabelsFromCSV(const std::string& csvPath) {
    std::vector<int> labels;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] �t�@�C�����J���܂���: " << csvPath << std::endl;
        return labels;  // ��̃x�N�g����Ԃ�
    }
    std::string line;
    while (std::getline(file, line)) {
        try {
            int label = std::stoi(line);
            labels.push_back(label);
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Invalid label: " << line << std::endl;
            labels.push_back(-1);
        }
    }
    if (labels.empty()) {
        std::cerr << "[DEBUG] ���x����1���ǂݍ��܂�܂���ł����B" << std::endl;
    }
	return labels;
}


// ���O������
void initLog(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_logFile.open(filename, std::ios::app);
    if (!g_logFile.is_open()) {
        std::cerr << "���O�t�@�C�����J���܂���: " << filename << std::endl;
    }
}


// ���O�o��
void log(const std::string& message, bool toConsole) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_logFile.is_open()) {
        g_logFile << message << std::endl;
    }
    if (toConsole) {
        std::cout << message << std::endl;
    }
}

// ���O�N���[�Y
void closeLog() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_logFile.is_open()) {
        g_logFile.close();
    }
}