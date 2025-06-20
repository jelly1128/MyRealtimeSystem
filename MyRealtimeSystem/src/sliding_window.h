#pragma once
#include <vector>

// スライディングウィンドウで主クラスをシングルラベル化
std::vector<int> slidingWindowToSingleLabel(
    const std::vector<std::vector<int>>& hardLabels,
    int windowSize = 11,
    int step = 1,
    int numMainClasses = 6
);