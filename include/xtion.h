#pragma once

// Local includes
#include "calibration.h"

// STL includes
#include <deque>
#include <string>
#include <memory>

// Ros includes
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

// OpenCV includes
#include <opencv2/opencv.hpp>

class Xtion {
public:
    Xtion(std::string name);

    std::string getName();

    std::string getFrameId();

    void addTopic(std::string topic);

    void setCalibration(Calibration c);

    Calibration& getCalibration();

    bool isComplete();

    void color_callback(const sensor_msgs::Image::ConstPtr &image);

    void depth_callback(const sensor_msgs::Image::ConstPtr &image);

    void subscribeCamera(image_transport::ImageTransport& _it);

    bool getUpToId(int id, std::vector<std::pair< int, cv::Mat > >& color, std::vector<std::pair< int, cv::Mat > >& depth);

    bool getIdAndClear(int id, std::pair< int, cv::Mat >& color, std::pair< int, cv::Mat >& depth);

    static std::string parseNameFromTopics(std::string topic);

private:
    std::string _name;
    std::string _frame_id;
    bool _frame_id_aquired;
    std::string _depth_topic;
    std::string _color_topic;
    bool _has_depth;
    bool _has_color;
    image_transport::Subscriber _subscriber_depth, _subscriber_color;
    std::deque< std::pair<int, cv::Mat> > _color_maps;
    std::deque< std::pair<int, cv::Mat> > _depth_maps;
    Calibration _calibration;
    bool _has_calibration;

    int _last_id;
};
