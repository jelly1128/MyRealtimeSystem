#pragma once
#include <string>

// ���u���o���f��
const std::string TREATMENT_MODEL_PATH = "models/treatment_model_fold_0_best.pt";

// ���悩��擾�ꍇ�̓���p�X
const std::string VIDEO_PATH = "D:/M1/����Ƃ�/000001-002.mov";

// �摜�t�H���_����擾�ꍇ�̃t�H���_�p�X
//const std::string VIDEO_FOLDER_PATH = "images/20211021093634_000001-003/";
//const std::string VIDEO_FOLDER_PATH = "images/20210524100043_000001-001/";
//const std::string VIDEO_FOLDER_PATH = "images/20210531112330_000001-001/";
//const std::string VIDEO_FOLDER_PATH = "images/debug/";
//const std::string VIDEO_NAME = "20211021093634_000001-001"; // ����t�@�C����
//const std::string VIDEO_NAME = "20211021093634_000001-003"; // ����t�@�C����
//const std::string VIDEO_NAME = "20210524100043_000001-001"; // ����t�@�C����
const std::string VIDEO_NAME = "20210531112330_000001-001"; // ����t�@�C����
const std::string VIDEO_FOLDER_PATH = std::string("images/") + VIDEO_NAME + "/"; // ����t�H���_�p�X


// �o�̓t�@�C����
const std::string OUTPUT_PROBS_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_probs.csv";
const std::string OUTPUT_LABELS_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_labels.csv";
const std::string OUTPUT_SMOOTHED_CSV = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_smoothed.csv";
const std::string TIMELINE_IMAGE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_timeline.png";
const std::string THUMBNAIL_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME;
const std::string LOG_FILE_PATH = std::string("outputs/") + VIDEO_NAME + "/" + VIDEO_NAME + "_log.txt";

// �萔��`
constexpr int NUM_CLASSES = 15;
constexpr int FRAME_INTERVAL = 10;

constexpr int INPUT_WIDTH = 224;
constexpr int INPUT_HEIGHT = 224;

constexpr int SLIDING_WINDOW_SIZE = 11; // Number of frames in the sliding window
constexpr int SLIDING_WINDOW_STEP = 1; // Step size for the sliding window
constexpr int NUM_SCENE_CLASSES = 6; // Number of main classes in the model