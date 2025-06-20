#include "sliding_window.h"
#include <algorithm>

std::vector<int> slidingWindowToSingleLabel(
    const std::vector<std::vector<int>>& hardLabels,
    int windowSize,
    int step,
    int numMainClasses
) {
    std::vector<int> singleLabels;
    int numFrames = hardLabels.size();

    int halfWin = windowSize / 2;
	
    for (int start = 0; start <= numFrames - windowSize; start += step) {
        std::vector<int> classCounts(numMainClasses, 0);

        for (int i = start; i < start + windowSize; ++i) {
            for (int c = 0; c < numMainClasses; ++c) {
                classCounts[c] += hardLabels[i][c];
            }
        }

        int maxIdx = std::distance(
            classCounts.begin(),
            std::max_element(classCounts.begin(), classCounts.end())
        );

        int centerFrame = start + halfWin;
        singleLabels.push_back(maxIdx);  // ���S�ɑΉ����郉�x���̂ݕۑ�
        // ���� centerFrame ���L�^�������ꍇ�͕ʃx�N�g���ɕۑ�
    }

    return singleLabels;
}
