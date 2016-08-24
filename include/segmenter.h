#pragma once

//Boost includes
#include <boost/thread/mutex.hpp>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

//fps mapper includes
#include <fps_ros_bridge/local_map_listener.h>
#include <fps_map/multi_image_map_node.h>
#include <core/multi_projector.h>

// Local includes
#include "calibration.h"
#include "feature_extractor.h"
#include "xtion.h"
#include "config.h"

// OpenCV includes
#include <opencv2/opencv.hpp>

// ROS includes
#include <image_transport/image_transport.h>
#include <ros/ros.h>

// Service includes
# include "semantic_segmentation/IdsSrv.h"
# include "semantic_segmentation/LocalMapSegmentationSrv.h"
# include "semantic_segmentation/SegmentationInformationSrv.h"

// STL includes
#include <vector>
#include <string>

// Third party includes
#include "json/json.h"

//libforest
#include "libforest/data.h"
#include "libforest/libforest.h"

class Segmenter : public fps_mapper::LocalMapListener{

public:
    Segmenter(ros::NodeHandle& n, std::string config_file, std::vector<std::string> topic_names, std::string base_link_name, bool external_semantics, bool debug_dump);

    ~Segmenter();

    void initializeProjector(fps_mapper::MultiImageMapNode* multi_image_node);

    void onNewNode(fps_mapper::MapNode* m);

    void onNewLocalMap(fps_mapper::LocalMap* lmap);

    void start();

    void processFramesFromQueueExternal();

    void processFramesFromQueueInternalRF();

    void processMapFromQueue();

    bool srvStoredSemanticsIds(semantic_segmentation::IdsSrv::Request& req, semantic_segmentation::IdsSrv::Response& resp);

    bool srvGetLocalMapSegmentation(semantic_segmentation::LocalMapSegmentationSrv::Request& req, semantic_segmentation::LocalMapSegmentationSrv::Response& resp);

    bool srvSegmentationInformation(semantic_segmentation::SegmentationInformationSrv::Request& req, semantic_segmentation::SegmentationInformationSrv::Response& resp);

private:
    //Subscription stuff
    ros::NodeHandle& _n;
    image_transport::ImageTransport _it;
    std::map<std::string, Xtion*> _camera_map;
    std::vector<std::string> _topics;
    std::string _base_link_name;

    //Service that segments the rgb-d frames.
    ros::ServiceClient _single_frame_segmentation_service;

    //Service for providing final segmentation data.
    ros::ServiceServer _local_map_ids_srv;
    ros::ServiceServer _get_local_map_segmentation_srv;
    ros::ServiceServer _semantic_information_srv;
    bool _external_semantics;

    //RF stuff in case we segment directly in the node.
    libf::RandomForest _random_forest;
    //Also we need a feature extractor.
    Features::FeatureExtractor _feature_extractor;

    //Used to "transfer" data between threads.
    boost::mutex _frame_mtx;
    boost::mutex _cloud_mtx;
    boost::mutex _cloud_processing_mtx;

    //storage stuff
    fps_mapper::MultiProjector _projector;
    std::vector<Xtion*> _cameras_in_order;
    unsigned int _camera_w;
    unsigned int _camera_h;
    bool _order_initialized;
    std::vector<std::deque< std::pair<int, std::pair<cv::Mat, cv::Mat> > > >    _image_queues;
    std::vector<std::deque< std::pair<int, std::vector<float> > > >             _result_queues;
    std::deque<fps_mapper::LocalMap*>                                           _local_map_queue;

    std::vector<std::pair<int, std::vector<std::vector<unsigned char> > > >     _cloud_results;

    //Segmentation stuff
    unsigned int                                             _layer_count;
    std::vector<std::string>                                 _layer_names;
    std::vector<unsigned int>                                _layer_class_counts;
    std::vector<std::vector<std::string> >                   _layer_class_names;
    std::vector<std::vector<std::vector<unsigned char> > >   _layer_class_colors;
    std::vector<unsigned char>                               _layer_unknown_label;

    //Flag to check if we will dump all the clouds to /tmp
    bool _dump_clouds_to_tmp;

    //For determining the differences between the current and the last frame.
    Eigen::Isometry3f _last_pose;
    int _last_key_frame_id;

    //Parameters loaded from the config file. Mainly determining the fine tuned behavior.
    bool _use_dense_crf;
    float _dcrf_xyz_kernel;
    float _dcrf_rgb_kernel;
    float _dcrf_kernel_weight;
    unsigned int _dcrf_iterations;

    unsigned int _rf_prediction_stride;

    float _depth_min;
    float _depth_max;

    float _keyframe_skip_rotation;
    float _keyframe_skip_translation;

};