#pragma once
#include <string>

const std::string TREATMENT_MODEL_PATH = "models/treatment_model_fold_0_best.pt";
const std::string VIDEO_PATH = "D:/M1/“®‰æ‚Æ‚©/000001-002.mov";
const std::string VIDEO_FOLDER_PATH = "images/20211021093634_000001-001/";
//const std::string VIDEO_FOLDER_PATH = "images/debug/";
const std::string OUTPUT_PROBS_CSV = "outputs/demo_output_probs.csv";
const std::string OUTPUT_LABELS_CSV = "outputs/demo_output_labels.csv";
const std::string OUTPUT_SMOOTHED_CSV = "outputs/demo_output_smoothed.csv";
const std::string TIMELINE_IMAGE_PATH = "outputs/timeline.png";

const int NUM_CLASSES = 15;
const int FRAME_INTERVAL = 10;

const int INPUT_WIDTH = 224;
const int INPUT_HEIGHT = 224;
