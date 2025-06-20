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
        singleLabels.push_back(maxIdx);  // 中心に対応するラベルのみ保存
        // もし centerFrame を記録したい場合は別ベクトルに保存
    }

    return singleLabels;
}
