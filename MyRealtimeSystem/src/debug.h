#pragma once
#include <vector>
#include <string>

const std::string DEBUG_PROBS_CSV = "outputs/debug/20211021093634_000001-001_probs.csv";
const std::string DEBUG_LABELS_CSV = "outputs/debug/20211021093634_000001-001_labels.csv";
const std::string DEBUG_SMOOTHED_CSV = "outputs/debug/20211021093634_000001-001_smoothed.csv";
const std::string DEBUG_LOG_FILE_PATH = "outputs/debug/debug_log.txt";


// CSVファイルからラベルを読み込む
std::vector<std::vector<float>> loadFrameProbabilitiesFromCSV(const std::string& csvPath);
std::vector<std::vector<int>> loadFrameBinariesFromCSV(const std::string& csvPath);
std::vector<int> loadWindowedSceneLabelsFromCSV(const std::string& csvPath);


// log
void initLog(const std::string& filename);
void log(const std::string& message, bool toConsole = true);
void closeLog();


// 構築中のアルゴリズム
//bool selectThumbnailsFromLabels();