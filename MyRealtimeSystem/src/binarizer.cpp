#include "binarizer.h"

// モデルの出力をバイナリ化する関数
std::vector<std::vector<int>> binarizeProbabilities(
    const std::vector<std::vector<float>>& probs,
    float threshold
) {
    std::vector<std::vector<int>> result;

    for (const auto& frame_probs : probs) {
        std::vector<int> binary_vec;
        for (float p : frame_probs) {
            binary_vec.push_back(p >= threshold ? 1 : 0);
        }
        result.push_back(binary_vec);
    }

    return result;
}
