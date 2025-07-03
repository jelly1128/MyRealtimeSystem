#pragma once
#include <vector>
#include <string>

const std::string DEBUG_PROBS_CSV = "outputs/debug/20211021093634_000001-001_probs.csv";
const std::string DEBUG_LABELS_CSV = "outputs/debug/20211021093634_000001-001_labels.csv";
const std::string DEBUG_SMOOTHED_CSV = "outputs/debug/20211021093634_000001-001_smoothed.csv";

// CSV�t�@�C�����烉�x����ǂݍ���
std::vector<std::vector<float>> loadFrameProbabilitiesFromCSV(const std::string& csvPath);
std::vector<std::vector<int>> loadFrameBinariesFromCSV(const std::string& csvPath);
std::vector<int> loadWindowedSceneLabelsFromCSV(const std::string& csvPath);


// log
void saveLogToFile(const std::string& filePath, const std::stringstream& logStream);


// �\�z���̃A���S���Y��
//bool selectThumbnailsFromLabels();