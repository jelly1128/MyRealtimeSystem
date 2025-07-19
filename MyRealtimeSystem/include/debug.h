#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>

const std::string DEBUG_TREATMENT_PROBS_CSV = "outputs/20220105102425_4/20220105102425_4_treatment_probs.csv";
const std::string DEBUG_LABELS_CSV = "outputs/debug/20211021093634_000001-001_labels.csv";
const std::string DEBUG_SMOOTHED_CSV = "outputs/debug/20211021093634_000001-001_smoothed.csv";
const std::string DEBUG_LOG_FILE_PATH = "outputs/debug/debug_log.txt";


// CSVファイルからラベルを読み込む
std::vector<std::vector<float>> loadTreatmentProbabilitiesFromCSV(const std::string& csvPath);
std::vector<int> loadSingleLabelsFromCSV(const std::string& csvPath);
std::vector<std::vector<int>> loadFrameBinariesFromCSV(const std::string& csvPath);


// log
void initLog(const std::string& filename);
void log(const std::string& message, bool toConsole = true);
void closeLog();

// 実行時間計測タイムロガークラス
class TimeLogger {
public:
    TimeLogger(const std::string& blockName, bool toConsole = true);
    void stop();
private:
    std::string blockName_;
    bool toConsole_;
    std::chrono::high_resolution_clock::time_point start_;
};