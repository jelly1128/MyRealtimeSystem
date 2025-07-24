#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// モデルのパス
const std::string TREATMENT_MODEL_PATH = "models/treatment_model_fold_0_best.pt";
//const std::string ORGAN_MODEL_PATH = "models/poc_model_v4_no_resize270_epoch50_BEST.pt";
const std::string ORGAN_MODEL_PATH = "models/umetsu_model.pt";

// 動画から取得場合の動画パス
const std::string VIDEO_PATH = "D:/M1/動画とか/000001-002.mov";


// 画像フォルダから取得場合のフォルダパス
//const std::string VIDEO_NAME = "20210524100043_000001-001"; // 動画ファイル名
//const std::string VIDEO_NAME = "20210531112330_000001-001"; // 動画ファイル名
//const std::string VIDEO_NAME = "20211021093634_000001-001"; // 動画ファイル名
const std::string VIDEO_NAME = "20211021093634_000001-003"; // 動画ファイル名
//const std::string VIDEO_NAME = "20220105102425_4"; // 動画ファイル名
const std::string VIDEO_FOLDER_PATH = std::string("images/") + VIDEO_NAME + "/"; // 動画フォルダパス

// ログ用のファイルパス
const std::string LOG_FILE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_treatment_log.txt";

// 処置検出用出力ファイル
const std::string TREATMENT_OUTPUT_PROBS_CSV        = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_treatment_probs.csv";
const std::string TREATMENT_OUTPUT_LABELS_CSV       = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_treatment_labels.csv";
const std::string TREATMENT_OUTPUT_SCENE_LABELS_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_treatment_scene_labels.csv";
const std::string TREATMENT_TIMELINE_IMAGE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_treatment_timeline.png";
const std::string HIGH_FREQ_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_high_frequency_scores.csv";

// 臓器分類用出力ファイル
const std::string ORGAN_OUTPUT_LABELS_CSV   = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_organ_labels.csv";
const std::string ORGAN_TIMELINE_IMAGE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_organ_timeline.png";

// 画像取得間隔（フレーム数）
constexpr int FRAME_INTERVAL = 10;

// 推論用画像のパラメータ
const std::string MASK_IMAGE_PATH = "images/fujifilm_mask.png"; // マスク画像のパス
const cv::Rect TREATMENT_CROP_BOX(330, 25, 1260, 970); // クロップボックス（x, y, width, height）
const cv::Rect ORGAN_CROP_BOX(320, 28, 1280, 1024); // クロップボックス（x, y, width, height）
constexpr int INPUT_WIDTH = 224;
constexpr int INPUT_HEIGHT = 224;
constexpr int ORGAN_INPUT_WIDTH = 270; // 臓器分類用の入力幅

// 処置検出
constexpr int NUM_CLASSES = 15;                     // 処置検出モデルのクラス数
constexpr int NUM_SCENE_CLASSES = 6;                // シーンクラスの数（0〜5の主クラス）
constexpr int TREATMENT_SLIDING_WINDOW_SIZE = 11;   // スライディングウィンドウ（多数決）の幅
constexpr int SLIDING_WINDOW_STEP = 1;              // スライディングウィンドウのステップサイズ
constexpr float BINARY_THRESHOLD = 0.5f;                // バイナリ化の閾値（0.5）

// 臓器分類
constexpr int ORGAN_SLIDING_WINDOW_SIZE = 60;       // スライディングウィンドウ（臓器境界探索）の幅

// タイムライン画像のパラメータ
constexpr int TIMELINE_IMAGE_WIDTH = 1000;          // タイムライン画像の幅
constexpr int TIMELINE_IMAGE_HEIGHT = 50;         // タイムライン画像の高さ

// サムネイル画像選定のパラメータ
const int THUMNAIL_FRAME_GAP = 15;
const int THUMNAIL_TOP_K = 20;
const std::string TREATMENT_THUMNAIL_IMAGE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_thumnail";

// ラベル種別定義
enum class OrganLabel {
	OUTSIDE = 0,           // 体外
	HEAD_NECK = 1,         // 頭頸部
	ESOPHAGUS = 2,         // 食道
	STOMACH = 3,           // 胃
	DUODENUM = 4,          // 十二指腸
};

enum class TreatmentLabel {
    WHITE = 0,             // 白色光
    LUGOL = 1,             // ルゴール
    INDIGO = 2,            // インディゴカルミン
    NBI_BLI = 3,           // 狭帯域光
    ROOM = 4,              // 室内
    BUCKET = 5,            // バケツ
    RESIDUE = 6,           // 残渣
    BLOOD = 7,             // 血液付着
    LIGHT_ARTIFACT = 8,    // 光アーチファクト
    SPLASH = 9,            // 粘膜洗浄（Splash）
    WALL_CONTACT = 10,     // 内壁接触
    TOO_BRIGHT = 11,       // 過度に明るい
    TOO_DARK = 12,         // 過度に暗い
    SPRAYER = 13,          // 散布剤の管
    BIOPSY_FORCEPS = 14    // 生検鉗子
};

// 各フレームにおける最終的なラベル情報（出力用）
struct FrameLabel {
    int frameIndex;             // フレーム番号
    int organLabel;             // 臓器ラベル（int型。OrganLabelに準拠）
    int treatmentSceneLabel;    // 処置シーンラベル（int型）
    int biopsyLabel;            // 生検ラベル（0 or 1）
};

