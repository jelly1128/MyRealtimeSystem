#include "thumbnail_selector.h"
#include "debug.h"
#include <algorithm>
#include <set>

SceneSegmentManager::SceneSegmentManager(int numLabels, int topK, int frameGap)
    : topK(topK), frameGap(frameGap) {
}

void SceneSegmentManager::update(int label, int frameIndex, const cv::Mat& frame, float dlScore, float hfScore) {
    ThumbnailCandidate candidate{ frameIndex, frame.clone(), dlScore, hfScore };

    // ラベルが変わった場合、前のラベルのセグメントを確定
    if (prevLabel != -1 && label != prevLabel) {
        auto& prevSeg = currentSegment[prevLabel];
        if (prevSeg.length > longestSegment[prevLabel].length) {
            longestSegment[prevLabel] = prevSeg;
        }
        currentSegment.erase(prevLabel);
    }

    // セグメント更新
    auto& seg = currentSegment[label];
    if (seg.length == 0) {
        seg.startFrameIndex = frameIndex;
    }
    seg.endFrameIndex = frameIndex;
    seg.length++;
    seg.topKThumbnails.push(candidate);
    if (seg.topKThumbnails.size() > topK) seg.topKThumbnails.pop();

    prevLabel = label;
}

void SceneSegmentManager::finalize() {
    for (const auto& [label, seg] : currentSegment) {
        if (seg.length > longestSegment[label].length) {
            longestSegment[label] = seg;
        }
    }
    currentSegment.clear();
}

std::map<int, std::vector<ThumbnailCandidate>> SceneSegmentManager::getFinalThumbnails() const {
    std::map<int, std::vector<ThumbnailCandidate>> result;
    for (const auto& [label, seg] : longestSegment) {
        result[label] = selectWithFrameGap(seg.topKThumbnails, frameGap, topK);
    }
    return result;
}

std::vector<ThumbnailCandidate> SceneSegmentManager::selectWithFrameGap(
    std::priority_queue<ThumbnailCandidate> candidates,
    int frameGap, int topK) const {

    std::vector<ThumbnailCandidate> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
    }

    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.combinedScore() > b.combinedScore();
        });

    std::vector<ThumbnailCandidate> selected;
    std::set<int> usedFrames;

    for (const auto& cand : sorted) {
        bool tooClose = false;
        for (int idx : usedFrames) {
            if (std::abs(cand.frameIndex - idx) < frameGap) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            selected.push_back(cand);
            usedFrames.insert(cand.frameIndex);
            if (selected.size() == topK) break;
        }
    }
    return selected;
}

void SceneSegmentManager::logSummary() const {
    log("=== ラベルごとの最長区間 ===", true);
    for (const auto& [label, seg] : longestSegment) {
        log("Label " + std::to_string(label)
            + " [" + std::to_string(seg.startFrameIndex) + "," + std::to_string(seg.endFrameIndex) + "]"
            + " (length=" + std::to_string(seg.length) + ")", true);

        std::priority_queue<ThumbnailCandidate> thumbs = seg.topKThumbnails;
        std::vector<ThumbnailCandidate> thumbsVec;
        while (!thumbs.empty()) {
            thumbsVec.push_back(thumbs.top());
            thumbs.pop();
        }
        std::sort(thumbsVec.begin(), thumbsVec.end(), [](const auto& a, const auto& b) {
            return a.combinedScore() > b.combinedScore();
            });
        for (const auto& cand : thumbsVec) {
            log("  Frame " + std::to_string(cand.frameIndex)
                + ", DLScore=" + std::to_string(cand.deepLearningScore)
                + ", HiFreq=" + std::to_string(cand.highFrequencyScore)
                + ", Score=" + std::to_string(cand.combinedScore()), true);
        }
    }
}
