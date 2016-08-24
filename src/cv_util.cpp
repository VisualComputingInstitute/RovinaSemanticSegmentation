#include "cv_util.h"

#include <stdexcept>
#include <string>
#include <fstream>
#include <ios>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

namespace Utils {

void SaveMat(const std::string& filename, const cv::Mat& data) {
    if (data.empty()) {
        throw std::runtime_error(std::string("No data was provided for saving to file: ") +filename);
    }

    std::ofstream out(filename.c_str(), std::ios::out|std::ios::binary);
    if (!out) {
        throw std::runtime_error(std::string("Could not create file: ") +filename);
    }

    int cols = data.cols;
    int rows = data.rows;
    int chan = data.channels();
    int eSiz = (data.dataend-data.datastart)/(cols*rows*chan);
    int type = data.type();

    // Write header
    out.write((char*)&cols,sizeof(cols));
    out.write((char*)&rows,sizeof(rows));
    out.write((char*)&chan,sizeof(chan));
    out.write((char*)&eSiz,sizeof(eSiz));
    out.write((char*)&type,sizeof(type));

    // Write data.
    if (data.isContinuous()) {
        out.write((char *)data.data,cols*rows*chan*eSiz);
    }
    else {
        throw std::runtime_error(std::string("Cannot write non-continuous data to file: ") +filename);
    }
    out.close();
}

void ReadMat(const std::string& filename, cv::Mat& data) {
    std::ifstream in(filename.c_str(), std::ios::in|std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("Could not open file: ") +filename);
    }
    int cols;
    int rows;
    int chan;
    int eSiz;
    int type;

    // Read header
    in.read((char*)&cols,sizeof(cols));
    in.read((char*)&rows,sizeof(rows));
    in.read((char*)&chan,sizeof(chan));
    in.read((char*)&eSiz,sizeof(eSiz));
    in.read((char*)&type,sizeof(type));

    // Alocate Matrix.
    data = cv::Mat(rows,cols,type);

    // Read data.
    if (data.isContinuous()) {
        in.read((char *)data.data,cols*rows*chan*eSiz);
    } else {
        throw std::runtime_error(std::string("Could not create a continuous cv::Mat to read from file: ") +filename);
    }
    in.close();
}

void ShowCvMat(const cv::Mat& m, std::string window_name) {
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( window_name, m );
    cv::waitKey(0);
}


//We compute a permutation of the bits. Here we assume we only get positive indices.
//Also, since it are indices, we only assume positive indices.
//Based on the above assumptions we have 2^24 (3*8 bits) indices ~ 8M
// This should suffice for most images in our case.
cv::Mat segmentIdToBgr(const cv::Mat& indices) {
    cv::Mat result(indices.rows, indices.cols, CV_8UC3);
    for(int y=0; y < indices.rows; ++y) {
        uchar*  color = result.ptr<uchar>(y);
        const int* index = indices.ptr<int>(y);
        for(int x=0; x < indices.cols; ++x) {
            uchar r =0;
            uchar g =0;
            uchar b =0;
            for(int i=0; i < 24; ++i) {
                uchar val = (*index & (1 << i)) >> i;
                if(i %3 == 0) {
                    r |= val << (7 - i/3);
                } else if( i % 3 == 1) {
                    g |= val << (7 - i/3);
                } else {
                    b |= val << (7 - i/3);
                }
            }
            color[0] = b;
            color[1] = g;
            color[2] = r;
            color+=3;
            index++;
        }
    }
    return result;
}
cv::Mat bgrToSegmentId(const cv::Mat& rgb) {
    cv::Mat result(rgb.rows, rgb.cols, CV_32SC1);
    for(int y=0; y < rgb.rows; ++y) {
        int*  index = result.ptr<int>(y);
        const uchar* color = rgb.ptr<uchar>(y);
        for(int x=0; x < rgb.cols; ++x) {
            uchar r =color[2];
            uchar g =color[1];
            uchar b =color[0];
            index[0] = 0;
            for(int i=0; i < 24; ++i) {
                int val = 0;
                if(i %3 == 0) {
                    val |= (r & (1 << (7 - i/3))) >> (7 - i/3);
                } else if( i % 3 == 1) {
                    val |= (g & (1 << (7 - i/3))) >> (7 - i/3);
                } else {
                    val |= (b & (1 << (7 - i/3))) >> (7 - i/3);
                }
                index[0] |= (val << i);
            }
            color+=3;
            index++;
        }
    }
    return result;
}


void ShowCvMatHeatMap(const cv::Mat& m, std::string window_name, int cm) {
    double min, max;
    cv::minMaxIdx(m, &min, &max);
    std::cout << "min value: " << min << " max value: "  << max << std::endl;
    cv::Mat temp;
    m.convertTo(temp,CV_8UC1, 255 / (max-min), -min*255 / (max-min));
    cv::applyColorMap(temp, temp, cm);

    ShowCvMat(temp, window_name);
}

}
