#pragma once
#include <vector>
#include <string>

// CSVファイルからラベルを読み込む
bool loadLabelsFromCSV(const std::string& csvPath, std::vector<int>& labels);
