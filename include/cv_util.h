#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <string>

namespace Utils{
void SaveMat(const std::string& filename, const cv::Mat& data);
void ReadMat(const std::string& filename, cv::Mat& data);

cv::Mat segmentIdToBgr(const cv::Mat& indices);
cv::Mat bgrToSegmentId(const cv::Mat& rgb);

void ShowCvMat(const cv::Mat& m, std::string window_name= std::string("OpenCV mat"));

void ShowCvMatHeatMap(const cv::Mat& m, std::string window_name= std::string("OpenCV mat"), int cm = cv::COLORMAP_JET );
}
