#pragma once
#include <vector>
#include <deque>
#include <optional>

/**
 * @brief スライディングウィンドウでシーンラベルを多数決抽出（同点は直前ラベル）
 *
 * @param treatmentLabels 全クラス（例：15クラス）のフレームごとのラベル配列（各要素はone-hot/multi-hot想定）
 * @param windowSize スライディングウィンドウのサイズ
 * @param step ウィンドウの移動ステップ
 * @param numMainClasses シーンクラス数（例：6なら0～5クラスが対象）
 * @return 各ウィンドウ中央フレームごとのシーンラベル配列（端フレームは出力しない）
 *
 * @details 各ウィンドウ内でシーンクラスごとの合計値を集計し、最多クラスをそのウィンドウ中央フレームのラベルとする。
 *          複数クラスが同数最多の場合は直前の出力ラベルを採用。最初のウィンドウのみ最小インデックスクラス。
 */
std::vector<int> slidingWindowExtractSceneLabels(
    const std::vector<std::vector<int>>& treatmentLabels,
    int windowSize,
    int step,
    int numSceneClasses
);


std::optional<int> processSceneLabelSlidingWindow(
    const std::deque<std::vector<int>>& windowSceneLabelBuffer,
    int prevSceneLabel
);