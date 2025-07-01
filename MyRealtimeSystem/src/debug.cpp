#include "debug.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

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
            if (colIndex > 0) {  // 2列目以降だけ読み込む
                try {
                    row.push_back(std::stof(value));
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number: " << value << std::endl;
                    row.push_back(0.0f);  // エラー処理（必要に応じて変更）
                }
            }
            ++colIndex;
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


bool loadMainLabelsFromCSV(const std::string& csvPath, std::vector<int>& labels) {
    labels.clear();
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] ファイルを開けません: " << csvPath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string filename, labelStr;
        // 1列目をスキップし、2列目を取得
        if (std::getline(ss, filename, ',') && std::getline(ss, labelStr, ',')) {
            try {
                int labelValue = std::stoi(labelStr);
                labels.push_back(labelValue);
            } catch (...) {
                std::cerr << "[DEBUG] 無効な値をスキップ(2列目): " << labelStr << std::endl;
                continue;
            }
        }
    }
    if (labels.empty()) {
        std::cerr << "[DEBUG] ラベルが1つも読み込まれませんでした。" << std::endl;
    }
    return !labels.empty();
}