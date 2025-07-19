//#include "thumbnail_selector.h"
//
//ThumbnailSelector::ThumbnailSelector(
//    int numSceneClasses,
//    int slidingWindowSize,
//    int topK,
//    int frameGap
//)
//    : numSceneClasses(numSceneClasses),
//    slidingWindowSize(slidingWindowSize),
//    topK(topK),
//    frameGap(frameGap),
//    halfWindowSize(slidingWindowSize / 2),
//    prevSceneLabel(-1)
//{
//}
//
//void ThumbnailSelector::pushFrame(
//    int frameIndex,
//    const cv::Mat& frame,
//    const std::vector<float>& treatmentProbabilities
//) {
//    // 実装内容...（省略：前のメッセージやプロトタイプ例参照）
//}
//
//std::map<int, std::vector<ThumbnailCandidate>> ThumbnailSelector::finalize() {
//    // 実装内容...（省略）
//}
