#include "debug.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

bool loadLabelsFromCSV(const std::string& csvPath, std::vector<int>& labels) {
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