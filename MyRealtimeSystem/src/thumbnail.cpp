#include "thumbnail.h"

double computeHighFrequencyEnergy(const cv::Mat& inputImg) {
    cv::Mat gray;
    if (inputImg.channels() == 3) {
        cv::cvtColor(inputImg, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = inputImg.clone();
    }

    gray.convertTo(gray, CV_32F);

    // DFT（複素数出力）
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

    // 高周波マスク作成（低周波中心を0に）
    int radius = std::min(gray.rows, gray.cols) / 8;
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
    cv::circle(mask, cv::Point(cx, cy), radius, cv::Scalar(0), -1);

    // エネルギー計算
    std::vector<cv::Mat> channels(2);
    cv::split(dftImg, channels);
    cv::Mat mag;
    cv::magnitude(channels[0], channels[1], mag);
    cv::pow(mag, 2, mag);  // エネルギー = 振幅^2

    double totalEnergy = cv::sum(mag)[0];

    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F);
    cv::Mat highFreqOnly;
    cv::multiply(mag, maskFloat, highFreqOnly);
    double highFreqEnergy = cv::sum(highFreqOnly)[0];

    return totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0.0;
}

