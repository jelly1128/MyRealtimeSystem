#include "thumbnail.h"


float computeHighFrequencyEnergy(const cv::Mat& inputImg) {
    cv::Mat gray;
    if (inputImg.channels() == 3) {
        cv::cvtColor(inputImg, gray, cv::COLOR_RGB2GRAY);
    }
    else {
        gray = inputImg.clone();
    }

    gray.convertTo(gray, CV_32F);

    // DFT�i���f���o�́j
    cv::Mat dftImg;
    cv::dft(gray, dftImg, cv::DFT_COMPLEX_OUTPUT);

    // FFTShift
    int cx = dftImg.cols / 2;
    int cy = dftImg.rows / 2;
    cv::Mat q0(dftImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(dftImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(dftImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(dftImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // �����g�}�X�N�쐬�i����g���S��0�Ɂj
    int radius = std::min(gray.rows, gray.cols) / 8;
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
    cv::circle(mask, cv::Point(cx, cy), radius, cv::Scalar(0), -1);

    // --- �}�X�N�i���S���l�p�`�ŃJ�b�g�j ---
    //int radius = std::min(gray.rows, gray.cols) / 8;
    //cv::Mat mask = cv::Mat::ones(gray.rows, gray.cols, CV_32F);
    //cv::Rect lowFreqBox(cx - radius, cy - radius, radius * 2, radius * 2);
    //mask(lowFreqBox) = 0.0f;  // ���S�̒���g�̈��0�ɂ���i�l�p�`�j

    // �G�l���M�[�v�Z
    std::vector<cv::Mat> channels(2);
    cv::split(dftImg, channels);
    cv::Mat mag;
    cv::magnitude(channels[0], channels[1], mag);
    cv::pow(mag, 2, mag);  // �G�l���M�[ = �U��^2

    double totalEnergy = cv::sum(mag)[0];

    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F);
    cv::Mat highFreqOnly;
    cv::multiply(mag, maskFloat, highFreqOnly);
    double highFreqEnergy = cv::sum(highFreqOnly)[0];

    return totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0.0;
}


std::vector<ThumbnailCandidate> selectThumbnailsWithFrameGap(
    std::priority_queue<ThumbnailCandidate> topKThumbs,
    int frameGap,
    int topK
) {
    std::vector<ThumbnailCandidate> allCandidates;
    while (!topKThumbs.empty()) {
        allCandidates.push_back(topKThumbs.top());
        topKThumbs.pop();
    }
    // �����ɍ~��sort
    std::sort(allCandidates.begin(), allCandidates.end(),
        [](const ThumbnailCandidate& a, const ThumbnailCandidate& b) {
            return a.combinedScore() > b.combinedScore();
        });

    std::vector<ThumbnailCandidate> selected;
    std::set<int> usedIndices;
    for (const auto& cand : allCandidates) {
        if (selected.size() >= topK) break;
        bool tooClose = false;
        for (int used : usedIndices) {
            if (std::abs(cand.frameIndex - used) < frameGap) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            selected.push_back(cand);
            usedIndices.insert(cand.frameIndex);
        }
    }
    return selected;
}