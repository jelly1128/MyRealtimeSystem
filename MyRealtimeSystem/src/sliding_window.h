#pragma once
#include <vector>

// �X���C�f�B���O�E�B���h�E�Ŏ�N���X���V���O�����x����
std::vector<int> slidingWindowToSingleLabel(
    const std::vector<std::vector<int>>& hardLabels,
    int windowSize = 11,
    int step = 1,
    int numMainClasses = 6
);