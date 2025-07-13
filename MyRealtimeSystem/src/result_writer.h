#pragma once
#include <vector>
#include <string>

// 予測結果（各フレームの確率）をCSV形式で保存
// Tは float や int に対応
template<typename T>
bool saveMatrixToCSV(
    const std::string& filename,
    const std::vector<std::vector<T>>& data,
    const std::string& prefix = "class_"
);

bool saveLabelsToCSV(
    const std::string& filename,
    const std::vector<int>& labels
);