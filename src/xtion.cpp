// local includes
#include "xtion.h"

// Ros includes
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <algorithm>

Xtion::Xtion(std::string name): _name(name){
    _has_depth = false;
    _has_color = false;
    _has_calibration = false;
    _last_id = 0;
    _frame_id_aquired = false;
}

std::string Xtion::getName(){
    return _name;
}

std::string Xtion::getFrameId(){
    if(_frame_id_aquired){
        return _frame_id;
    }else{
        throw std::runtime_error("Camera " + _name + " did not receive any messages yet, so a frame id could not be determined.");
    }
}

void Xtion::addTopic(std::string topic){
    //Check if this is rgb (or color) / depth.
    if(topic.find("rgb") != std::string::npos || topic.find("color") != std::string::npos){
        if(!_has_color){
            _color_topic = topic;
            _has_color = true;
        }else{
            throw std::runtime_error("Camera " + _name + " already has the color topic: " + _color_topic + " but :" + topic + " should be added!");
        }
    }else if(topic.find("depth") != std::string::npos){
        if(!_has_depth){
            _depth_topic = topic;
            _has_depth = true;
        }else{
            throw std::runtime_error("Camera " + _name + " already has the color topic: " + _depth_topic + " but :" + topic + " should be added!");
        }
    }else{
        throw std::runtime_error("Missformed topic name: " + topic + " found");
    }
}

void Xtion::setCalibration(Calibration c){
    _calibration = c;
    _has_calibration= true;
}

Calibration& Xtion::getCalibration(){
    if( !_has_calibration){
        throw std::runtime_error("Camera " + _name + " has no calibration yet.");
    }else{
        return _calibration;
    }
}

bool Xtion::isComplete(){
    return _has_color && _has_depth;
}

void Xtion::color_callback(const sensor_msgs::Image::ConstPtr &image){
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(image);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat temp(cv_ptr->image);
    _color_maps.push_back(std::pair<int, cv::Mat>(image->header.seq, temp.clone()));
    if(_frame_id_aquired == false){
        _frame_id = image->header.frame_id;
        _frame_id_aquired = true;
    }
}

void Xtion::depth_callback(const sensor_msgs::Image::ConstPtr &image){
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(image);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat temp(cv_ptr->image);
    _depth_maps.push_back(std::pair<int, cv::Mat>(image->header.seq, temp.clone()));
}

void Xtion::subscribeCamera(image_transport::ImageTransport& _it){
    if(isComplete()){
        _subscriber_color = _it.subscribe(_color_topic, 100, &Xtion::color_callback, this);
        _subscriber_depth = _it.subscribe(_depth_topic, 100, &Xtion::depth_callback, this);
    }else{
        throw std::runtime_error("Camera " + _name + " did not have both color and depth topic, so subscribing failed!");
    }
}

bool Xtion::getUpToId(int id, std::vector<std::pair< int, cv::Mat > >& color, std::vector<std::pair< int, cv::Mat > >& depth){
    if(id < _color_maps.front().first){
        //We no longer have (or never had) this id.
        return false;
    }
    if(id > std::min(_color_maps.back().first, _depth_maps.back().first)){
        throw std::runtime_error("Requested id is not even available yet!");
    }else{ // Okay, we can work with this
        
        int available_id;
        //As some messages might be dropped, the lists don't need to align. We need to get the depth and color separately :(
        available_id = _color_maps.front().first;
        while (available_id <= id){
            color.push_back(_color_maps.front());
            _color_maps.pop_front();
            available_id = _color_maps.front().first;
        }
        available_id = _depth_maps.front().first;
        while (available_id <= id){
            depth.push_back(_depth_maps.front());
            _depth_maps.pop_front();
            available_id = _depth_maps.front().first;
        }
        return true;
    }
}

bool Xtion::getIdAndClear(int id, std::pair< int, cv::Mat >& color, std::pair< int, cv::Mat >& depth){
    // Check if the requested id is that of an old image.
    if(id < _last_id){
        _last_id = std::max(_last_id, id);
        return false;
    }
    if(id > std::min(_color_maps.back().first, _depth_maps.back().first)){
        throw std::runtime_error("Requested id is not even available yet!");
    }else{ // Okay, we can work with this
        int available_id;
        //As some messages might be dropped, the lists don't need to align. We need to get the depth and color separately :(
        available_id = _color_maps.front().first;
        while (available_id < id){
            _color_maps.pop_front();
            available_id = _color_maps.front().first;
        }
        color = _color_maps.front();
        _color_maps.pop_front();
        available_id = _depth_maps.front().first;
        while (available_id < id){
            _depth_maps.pop_front();
            available_id = _depth_maps.front().first;
        }
        depth = _depth_maps.front();
        _depth_maps.pop_front();
        _last_id = std::max(_last_id, id);
        return true;
    }
}

std::string Xtion::parseNameFromTopics(std::string topic){
    size_t pos = topic.find("/", 1);  //skip the first, as it is a slash, then go find the second. 
    std::string id = topic.substr(1, pos-1);
    //This is a bit dirty. Now some topics start with /ban/camera....
    //so the first part of the topic cannot be used as the unique identifier.
    //We assume that the second occurance of the string should be fine.
    if (pos < 8){ //This does not contain camera
        pos = topic.find("/", pos+1);
    }
    return topic.substr(1, pos-1);
}