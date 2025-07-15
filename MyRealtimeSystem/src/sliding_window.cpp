#include "sliding_window.h"
#include <algorithm>

std::vector<int> slidingWindowExtractSceneLabels(
    const std::vector<std::vector<int>>& treatmentLabels,
    int windowSize,
    int step,
    int numSceneClasses
) {
    // numSceneClassesの（0〜5）のシーンクラスのみを抽出した新たな配列を作成
    std::vector<std::vector<int>> sceneLabels;
    for (const std::vector<int>& vec : treatmentLabels) {
        // 各フレームについて、先頭numMainClasses分だけを抽出
        sceneLabels.emplace_back(vec.begin(), vec.begin() + numSceneClasses);
    }

    std::vector<int> sceneSingleLabels; // 出力結果（中央フレームごとのシーンラベル）
    int numFrames = sceneLabels.size();
    int halfWin = windowSize / 2;      // ウィンドウの中央位置

    int prevLabel = -1; // 直前のラベルを保持（同点多数決時に使用、最初は未定義）
	
    // スライディングウィンドウを左から右にずらしていく
    for (int start = 0; start <= numFrames - windowSize; start += step) {
        std::vector<int> classCounts(numSceneClasses, 0);

        // 現ウィンドウ内で各クラスの合計値を計算
        for (int i = start; i < start + windowSize; ++i) {
            for (int c = 0; c < numSceneClasses; ++c) {
                classCounts[c] += sceneLabels[i][c];
            }
        }

        // 最も多く出現したクラスのインデックスを抽出（複数あればすべて取得）
        int maxCount = *std::max_element(classCounts.begin(), classCounts.end());
        std::vector<int> maxIndices;
        for (int c = 0; c < numSceneClasses; ++c) {
            if (classCounts[c] == maxCount) maxIndices.push_back(c);
        }

        int label;
        if (maxIndices.size() == 1) {
            // 単独最大のクラスがあればそれを採用
            label = maxIndices[0];
        }
        else if (!sceneSingleLabels.empty()) {
            // 同点の場合は直前のラベルを再利用
            label = sceneSingleLabels.back();
        }
        else {
            // 最初のウィンドウのみ、直前ラベルが無いので最小インデックスのクラスを採用
            label = maxIndices[0];
        }

        // 中央フレームに決定したラベルを格納
        sceneSingleLabels.push_back(label);
    }

    return sceneSingleLabels;
}
