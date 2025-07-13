#pragma once
#include <vector>
#include <string>

// �\�����ʁi�e�t���[���̊m���j��CSV�`���ŕۑ�
// T�� float �� int �ɑΉ�
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