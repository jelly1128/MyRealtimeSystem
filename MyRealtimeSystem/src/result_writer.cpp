#include "result_writer.h"
#include <fstream>
#include <iomanip>

bool saveProbabilitiesToCSV(const std::string& filename, const std::vector<std::vector<float>>& allProbs) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        return false;
    }

    // ヘッダー行
    ofs << "frame";
    if (!allProbs.empty()) {
        for (int i = 0; i < allProbs[0].size(); ++i) {
            ofs << ",class_" << i;
        }
    }
    ofs << "\n";

    for (size_t i = 0; i < allProbs.size(); ++i) {
        ofs << i;
        for (float val : allProbs[i]) {
            ofs << "," << std::fixed << std::setprecision(6) << val;
        }
        ofs << "\n";
    }

    ofs.close();
    return true;
}
