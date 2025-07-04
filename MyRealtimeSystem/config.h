#pragma once
#include <string>

// 処置検出モデル
const std::string TREATMENT_MODEL_PATH = "models/treatment_model_fold_0_best.pt";

// 動画から取得場合の動画パス
const std::string VIDEO_PATH = "D:/M1/動画とか/000001-002.mov";

// 画像フォルダから取得場合のフォルダパス
//const std::string VIDEO_FOLDER_PATH = "images/20211021093634_000001-003/";
//const std::string VIDEO_FOLDER_PATH = "images/20210524100043_000001-001/";
//const std::string VIDEO_FOLDER_PATH = "images/20210531112330_000001-001/";
//const std::string VIDEO_FOLDER_PATH = "images/debug/";
//const std::string VIDEO_NAME = "20211021093634_000001-001"; // 動画ファイル名
//const std::string VIDEO_NAME = "20211021093634_000001-003"; // 動画ファイル名
//const std::string VIDEO_NAME = "20210524100043_000001-001"; // 動画ファイル名
const std::string VIDEO_NAME = "20210531112330_000001-001"; // 動画ファイル名
const std::string VIDEO_FOLDER_PATH = std::string("images/") + VIDEO_NAME + "/"; // 動画フォルダパス


// 出力ファイル名
const std::string OUTPUT_PROBS_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_probs.csv";
const std::string OUTPUT_LABELS_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_labels.csv";
const std::string OUTPUT_SMOOTHED_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_smoothed.csv";
const std::string TIMELINE_IMAGE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_timeline.png";
const std::string THUMBNAIL_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME;
const std::string LOG_FILE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_log.txt";

// 定数定義
constexpr int NUM_CLASSES = 15;
constexpr int FRAME_INTERVAL = 10;

constexpr int INPUT_WIDTH = 224;
constexpr int INPUT_HEIGHT = 224;

constexpr int SLIDING_WINDOW_SIZE = 11; // Number of frames in the sliding window
constexpr int SLIDING_WINDOW_STEP = 1; // Step size for the sliding window
constexpr int NUM_SCENE_CLASSES = 6; // Number of main classes in the model