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

// CSVファイルからフレームの確率を読み込む関数
//使い方
// for debug
// std::vector<std::vector<float>> frameProbabilities;
// frameProbabilities = loadFrameProbabilitiesFromCSV(DEBUG_PROBS_CSV);
std::vector<std::vector<float>> loadFrameProbabilitiesFromCSV(const std::string& csvPath) {
    std::vector<std::vector<float>> probabilities;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] ファイルを開けません: " << csvPath << std::endl;
        return probabilities;  // 空のベクトルを返す
    }
    std::string line;
    // 一行目（ヘッダー）をスキップ
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
                row.push_back(-1.0);  // エラー処理（必要に応じて変更）
            }
        }
        if (!row.empty()) {
            probabilities.push_back(row);
		}
    }
    if (probabilities.empty()) {
        std::cerr << "[DEBUG] 確率が1つも読み込まれませんでした。" << std::endl;
    }
	return probabilities;
}


std::vector<std::vector<int>> loadFrameBinariesFromCSV(const std::string& csvPath) {
    std::vector<std::vector<int>> binaries;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] ファイルを開けません: " << csvPath << std::endl;
        return binaries;  // 空のベクトルを返す
    }
    std::string line;
    // 一行目（ヘッダー）をスキップ
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
        std::cerr << "[DEBUG] バイナリが1つも読み込まれませんでした。" << std::endl;
    }
	return binaries;
}


std::vector<int> loadWindowedSceneLabelsFromCSV(const std::string& csvPath) {
    std::vector<int> labels;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] ファイルを開けません: " << csvPath << std::endl;
        return labels;  // 空のベクトルを返す
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
        std::cerr << "[DEBUG] ラベルが1つも読み込まれませんでした。" << std::endl;
    }
	return labels;
}


// ログ初期化
void initLog(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_logFile.open(filename, std::ios::app);
    if (!g_logFile.is_open()) {
        std::cerr << "ログファイルを開けません: " << filename << std::endl;
    }
}


// ログ出力
void log(const std::string& message, bool toConsole) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_logFile.is_open()) {
        g_logFile << message << std::endl;
    }
    if (toConsole) {
        std::cout << message << std::endl;
    }
}

// ログクローズ
void closeLog() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_logFile.is_open()) {
        g_logFile.close();
    }
}