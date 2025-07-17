#include "binarizer.h"

// ���f���̏o�͂��o�C�i��������֐�
std::vector<int> binarizeProbabilities(
    const std::vector<float>& probs,
    float threshold
) {
    std::vector<int> binary_vec;
    for (float p : probs) {
        binary_vec.push_back(p >= threshold ? 1 : 0);
    }
    return binary_vec;
}