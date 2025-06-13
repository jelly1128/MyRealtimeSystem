#include <fstream>
#include <iomanip>
#include "result_writer.h"

template<typename T>
bool saveMatrixToCSV(const std::string& filename,
                     const std::vector<std::vector<T>>& data,
                     const std::string& prefix) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return false;

    // �w�b�_�[
    ofs << "frame";
    if (!data.empty()) {
        for (int i = 0; i < data[0].size(); ++i) {
            ofs << "," << prefix << i;
        }
    }
    ofs << "\n";

    for (size_t i = 0; i < data.size(); ++i) {
        ofs << i;
        for (const auto& val : data[i]) {
            if constexpr (std::is_floating_point<T>::value) {
                ofs << "," << std::fixed << std::setprecision(6) << val;
            } else {
                ofs << "," << val;
            }
        }
        ofs << "\n";
    }

    ofs.close();
    return true;
}

// �����I�ȃC���X�^���X���icpp���Ŏg�������ꍇ�j
template bool saveMatrixToCSV<float>(const std::string&, const std::vector<std::vector<float>>&, const std::string&);
template bool saveMatrixToCSV<int>(const std::string&, const std::vector<std::vector<int>>&, const std::string&);
