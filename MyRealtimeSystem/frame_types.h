#pragma once
#include <vector>

// 各フレームにおける最終的なラベル情報（出力用）
struct FrameLabel {
    int frameIndex;             // フレーム番号
    int organLabel;             // 臓器ラベル（int型。OrganLabelに準拠）
    int treatmentSceneLabel;    // 処置シーンラベル（int型）
    int biopsyLabel;            // 生検ラベル（0 or 1）
};

// サムネイル選定やスコア処理用のスコア付き構造体（処理中間用）
struct FrameData {
    int frameIndex;                     // フレーム番号
    std::vector<float> probabilities;   // 推論スコア（15クラス）
    int swLabel;                        // 平滑化されたラベル
    float S_target = 0.0f;              // ターゲットスコア
    float S_event = 0.0f;               // イベントスコア
};