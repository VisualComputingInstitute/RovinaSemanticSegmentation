// Local includes
#include "xtion.h"
#include "segmenter.h"

// ROS includes
#include <ros/ros.h>

// STL includes
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    ros::init(argc, argv, "semantic_segmentation");
    std::string ns = ros::this_node::getName();
    ns += "/";
    ros::NodeHandle n;

    std::string config_file;
    n.getParam(ns+"config_file", config_file);

    std::vector<std::string> topic_param_names = {
        "t_color_1",
        "t_depth_1",
        "t_color_2",
        "t_depth_2",
        "t_color_3",
        "t_depth_3"
    };
    std::vector<std::string> topic_names;
    for(unsigned int i = 0; i < topic_param_names.size(); i++){
        std::string tmp;
        n.getParam(ns+topic_param_names[i], tmp);
        if(tmp.compare("")){
            topic_names.push_back(tmp);
        }
    }

    std::string base_link;
    n.getParam(ns+"base_link_frame_id", base_link);

    bool use_external_segmentation_service;
    n.getParam(ns+"external_semantics", use_external_segmentation_service);

    bool debug_dump;
    n.getParam(ns+"dump_clouds_to_tmp", debug_dump);

    Segmenter d(n, config_file, topic_names, base_link, use_external_segmentation_service, debug_dump);
    d.start();
    ros::spin();
}
