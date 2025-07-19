//#pragma once
//
//#include <vector>
//#include <deque>
//#include <map>
//#include <opencv2/opencv.hpp>
//
//#include "../config.h"       // FrameDataなど汎用型
//#include "thumbnail.h"       // ThumbnailCandidate, VideoSegment
//#include "sliding_window.h"  // processSceneLabelSlidingWindowなど
//#include "binarizer.h"         // binarizeProbabilities
//
//class ThumbnailSelector {
//private:
//    int numSceneClasses;
//    int slidingWindowSize;
//    int topK;
//    int frameGap;
//    int halfWindowSize;
//
//    std::deque<std::vector<int>> windowSceneLabelBuffer;
//    std::unordered_map<int, cv::Mat> windowFrameBuffer;
//    std::deque<int> windowIndices;
//    int prevSceneLabel;
//    std::map<int, VideoSegment> currentSegment;
//    std::map<int, VideoSegment> longestSegment;
//
//public:
//    ThumbnailSelector(
//        int numSceneClasses,
//        int slidingWindowSize,
//        int topK,
//        int frameGap
//    );
//
//    // フレームごとに呼ぶ
//    void pushFrame(int frameIndex, const cv::Mat& frame, const std::vector<float>& treatmentProbabilities);
//
//    // 検査終了時に呼ぶ
//    std::map<int, std::vector<ThumbnailCandidate>> finalize();
//
//};
