#pragma once
#include <vector>
#include <string>

const std::string DEBUG_PROBS_CSV = "outputs/debug/demo_output_probs.csv";
const std::string DEBUG_LABELS_CSV = "outputs/debug/demo_output_labels";

// CSVファイルからラベルを読み込む
std::vector<std::vector<float>> loadFrameProbabilitiesFromCSV(const std::string& csvPath);
bool loadMainLabelsFromCSV(const std::string& csvPath, std::vector<int>& labels);
