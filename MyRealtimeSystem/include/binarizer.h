#pragma once
#include <vector>

// 確率をしきい値で2値化する関数
std::vector<int> binarizeProbabilities(
    const std::vector<float>& probs,
    float threshold = 0.5
);
