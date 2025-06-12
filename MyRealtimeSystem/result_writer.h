#pragma once
#include <vector>
#include <string>

// 予測結果（各フレームの確率）をCSV形式で保存
bool saveProbabilitiesToCSV(const std::string& filename, const std::vector<std::vector<float>>& allProbs);
