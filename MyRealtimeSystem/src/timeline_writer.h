#pragma once
#pragma once
#include <vector>
#include <string>

// �^�C�����C���摜��`�悵��PNG�Ƃ��ĕۑ�
bool drawTimelineImage(
    const std::vector<int>& labels,
    const std::string& savePath,
    int numClasses = 6,
    int frameWidth = 4,
    int rowHeight = 50
);
