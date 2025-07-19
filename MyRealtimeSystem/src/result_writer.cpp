#include "result_writer.h"

template<typename T>
bool saveMatrixToCSV(const std::string& filename,
                     const std::vector<std::vector<T>>& data,
                     const std::string& prefix) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return false;

    // ヘッダー
    if (!data.empty()) {
        for (int i = 0; i < data[0].size(); ++i) {
            ofs <<  prefix << i;
            if (i < data[0].size() - 1) {
                ofs << ",";
			}
        }
    }
    ofs << "\n";

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            if (j > 0) ofs << ",";
            if constexpr (std::is_floating_point<T>::value) {
                ofs << std::fixed << std::setprecision(3) << data[i][j];
            }
            else {
                ofs << data[i][j];
            }
        }
        ofs << "\n";
    }

    ofs.close();
    return true;
}

// ここで明示的にインスタンス化
template bool saveMatrixToCSV<int>(const std::string&, const std::vector<std::vector<int>>&, const std::string&);
template bool saveMatrixToCSV<float>(const std::string&, const std::vector<std::vector<float>>&, const std::string&);

// 1次元ラベルベクトルをCSV保存
bool saveLabelsToCSV(const std::string& filename, const std::vector<int>& labels) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return false;
    for (size_t i = 0; i < labels.size(); ++i) {
        ofs << labels[i] << "\n";
    }
    return true;
}