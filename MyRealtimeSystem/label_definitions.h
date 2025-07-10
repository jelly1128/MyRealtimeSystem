#pragma once

// シーン、処置、生検の各種ラベルID定義
// 臓器ラベル（scene）
enum class OrganLabel {
	OUTSIDE_BODY = 0, // 体外
	HEAD_NECK = 1,    // 頭頸部
	ESOPHAGUS = 2,    // 食道
	STOMACH = 3,      // 胃
	DUODENUM = 4      // 十二指腸
};

// 処置ラベル（treatment scene）
enum class TreatmentLabel {
    WHITE = 0,        // 白色光
	LUGOL = 1,        // ルゴール
    INDIGO = 2,       // インディゴカルミン
	NBI = 3,          // NBI（狭帯域光）
	INSIDE_ROOM = 4,  // 室内
	BUCKET = 5,       // バケツ
};

// 生検鉗子ラベル（eventの一種）
// BIOPSY = 1, その他 = 0 で処理
constexpr int BIOPSY_PRESENT = 1;
constexpr int BIOPSY_ABSENT = 0;

