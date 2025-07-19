#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm>
#include <string>
#include <vector>


// �^�C�����C���摜��`�悵��PNG�Ƃ��ĕۑ�
bool drawTimelineImage(
    const std::vector<int>& labels,
    const std::string& savePath,
    int numSceneClasses,
    int timelineWidth = 1200,
    int timelineHeight = 35
);
