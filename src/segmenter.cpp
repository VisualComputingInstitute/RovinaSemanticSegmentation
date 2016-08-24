//Boost includes
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

//fps mapper includes
#include <core/multi_projector.h>

// Local includes
#include "segmenter.h"
#include "config.h"
#include "cv_util.h"

// Ros includes
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_listener.h>

// STL includes
#include <chrono>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <thread>

// Service includes
# include "semantic_segmentation/SingleFrameSegmentation.h"

// Third party includes
#include "json/json.h"
#include "densecrf.h"

// ROS includes
#include <ros/package.h>

Segmenter::Segmenter(ros::NodeHandle& n, std::string config_file, std::vector<std::string> topic_names,
                     std::string base_link_name, bool external_semantics,
                     bool debug_dump):
            _n(n), _it(image_transport::ImageTransport(n)), _topics(topic_names),
            _base_link_name(base_link_name),
            _external_semantics(external_semantics), _order_initialized(false),
            _dump_clouds_to_tmp(debug_dump){

    //Check if there are always depth and rgb topic pairs.
    //Assume the first part of the topic is the camera name, this should be unique.
    for(std::vector<std::string>::iterator topic_it = _topics.begin(); topic_it != _topics.end(); ++topic_it){
        //Get the camera name from the topic.
        std::string camera_name = Xtion::parseNameFromTopics(*topic_it);
        //Make sure the camera object exists.
        if(_camera_map.count(camera_name) == 0){
            _camera_map[camera_name] = new Xtion(camera_name);
        }
        //Add the topic to the camera.
        _camera_map[camera_name]-> addTopic(*topic_it);
    }
    //check if all cameras are ok, as in all have exactly one depth and color topic.
    bool cameras_ok = true;
    for(std::map<std::string, Xtion*>::iterator it = _camera_map.begin(); it != _camera_map.end(); ++it){
        cameras_ok &= it->second->isComplete();
    }
    //If not we cannot run. 
    if(!cameras_ok){
        throw std::runtime_error("cannot match rgb and depth pairs from the provided topics!");
    }
    _camera_w = 0;
    _camera_h = 0;

    Utils::Config conf(config_file, std::map<std::string, std::string>());

    //Parse the config to get all the labeling information.
    Json::Value coding_list = conf.getRaw("color_codings");
    _layer_count = coding_list.size();
    for(auto l :coding_list){
        _layer_names.push_back(l["name"].asString());
        Json::Value classes = l["coding"];
        _layer_class_names.push_back(std::vector<std::string>());
        _layer_class_colors.push_back(std::vector<std::vector<unsigned char> >());
        for(auto c :classes){
            if(c["label"].asInt()  >= 0){
                _layer_class_names.back().push_back(c["name"].asString());
                _layer_class_colors.back().push_back(std::vector<unsigned char>());
                _layer_class_colors.back().back().push_back(c["color"][0].asUInt());
                _layer_class_colors.back().back().push_back(c["color"][1].asUInt());
                _layer_class_colors.back().back().push_back(c["color"][2].asUInt());
            }
            //We look for the Unknown label so we can use this as the default label.
            if(c["name"].asString().compare("Unknown") == 0){
                _layer_unknown_label.push_back(_layer_class_names.back().size() -1);
            }
        }
        //Check if we found an unknown label, otherwise just set it to zero. This will result in no special behaviour.
        if(_layer_names.size() != _layer_unknown_label.size()){
            _layer_unknown_label.push_back(0);
        }
        _layer_class_counts.push_back(_layer_class_names.back().size());
    }

    //Outsource the labeling?
    if(_external_semantics){
        //Subscribe to the service for labeling
        _single_frame_segmentation_service = n.serviceClient<semantic_segmentation::SingleFrameSegmentation>("/semantic_segmentation/SingleFrameSegmentation");
    }else{
        //We'll load a random forest model that will to the segmentation.
        std::string model_file = ros::package::getPath("semantic_segmentation") + std::string("/resources/forest.dat");
        libf::RandomForest* tmp = new libf::RandomForest();

        std::filebuf fb;
        if (fb.open (model_file,std::ios::in)){
          std::istream is(&fb);
          tmp->read(is);
        }
        fb.close();
        _random_forest = *tmp;
        _feature_extractor = Features::FeatureExtractor(conf);
    }

    //Load params from the config file
    _use_dense_crf = conf.get<bool>("use_dense_crf");
    _dcrf_xyz_kernel = conf.get<float>("dcrf_xyz_kernel");
    _dcrf_rgb_kernel = conf.get<float>("dcrf_rgb_kernel");
    _dcrf_kernel_weight = conf.get<float>("dcrf_kernel_weight");
    _dcrf_iterations = conf.get<unsigned int>("dcrf_iterations");
    _rf_prediction_stride = conf.get<unsigned int>("rf_prediction_stride");
    _depth_min = conf.get<float>("depth_min");
    _depth_max = conf.get<float>("depth_max");
    _keyframe_skip_rotation = conf.get<float>("keyframe_skip_rotation");
    _keyframe_skip_translation = conf.get<float>("keyframe_skip_translation");

    //Initialize the last pose and keyframe;
    _last_pose.linear() = Eigen::Matrix3f::Identity();
    _last_pose.translation() = Eigen::Vector3f::Ones()*10; //Far far away to make sure we always take the first frame.
    _last_key_frame_id = 0;
}

Segmenter::~Segmenter(){
    for(std::map<std::string, Xtion*>::iterator it = _camera_map.begin(); it != _camera_map.end(); ++it){
        delete it->second;
    }
}

//Parses the cameras and camera order from a multi image map node.
void Segmenter::initializeProjector(fps_mapper::MultiImageMapNode* multi_image_node){
    tf::TransformListener listener;
    _frame_mtx.lock(); //This is stupid, should be removed, but makes everything a little more stable here. 
    std::vector<fps_mapper::BaseCameraInfo*>& camera_infos = multi_image_node->cameraInfo()->cameraInfos();
    for(unsigned int i = 0; i < camera_infos.size(); ++i){
        if(camera_infos[i]){
            std::cout << camera_infos[i]->topic() << std::endl;
        }else{
            std::cerr << "Not there yet, nr: " << i << std::endl;
        }
        std::string name = Xtion::parseNameFromTopics(camera_infos[i]->topic());
        if(_camera_map.count(name) != 0){
            _cameras_in_order.push_back(_camera_map[name]);
            _image_queues.push_back(std::deque< std::pair<int, std::pair<cv::Mat, cv::Mat> > >());
            _result_queues.push_back(std::deque< std::pair<int, std::vector<float> > >());

            //Create the calibration for this one.
            Calibration c;
            c._intrinsic = dynamic_cast<fps_mapper::PinholeCameraInfo*>(camera_infos[i])->cameraMatrix();
            c._intrinsic_inverse = c._intrinsic.inverse();

            //To get the extrinsics we need the camera frame_id. Since we can match the topic from the fps_mapper
            //we can use the frame id from that camera info.
            std::string camera_frame_id = _cameras_in_order.back()->getFrameId();

            tf::StampedTransform transform;
            try{
            listener.waitForTransform(_base_link_name,
                        camera_frame_id,
                        ros::Time(0),
                        ros::Duration(10) );
            listener.lookupTransform (_base_link_name,
                        camera_frame_id,
                        ros::Time(0),
                        transform);
            }catch (tf::TransformException ex){
                ROS_ERROR("%s",ex.what());
            }
            Eigen::Quaternionf q;
            q.x() = transform.getRotation().x();
            q.y() = transform.getRotation().y();
            q.z() = transform.getRotation().z();
            q.w() = transform.getRotation().w();
            c._extrinsic.linear()=q.toRotationMatrix();
            c._extrinsic.translation()=Eigen::Vector3f(transform.getOrigin().x(),
                                                       transform.getOrigin().y(),
                                                       transform.getOrigin().z());

            _cameras_in_order.back()->setCalibration(c);

            if(_camera_w == 0 && _camera_h ==0){
                _camera_w = c._intrinsic(6)*2;
                _camera_h = c._intrinsic(7)*2;
            }else if(_camera_w != c._intrinsic(6)*2 || _camera_h != c._intrinsic(7)*2){
                throw std::runtime_error("Cameras with different image sizes are not supported!");
            }
            

        }else{
            throw std::runtime_error("Unknown camera found in map node: " +name );
        }
    }

    //Delete all the other cameras
    for(auto c_mapped : _camera_map){
        Xtion* current = c_mapped.second;
        bool used = false;
        for(auto c : _cameras_in_order){
            if(current->getName().compare(c->getName()) == 0 ){
                used = true;
                break;
            }
        }
        if(!used){
            std::cout << "Not using camera " << current->getName() << std::endl;
            delete current;
            _camera_map.erase(c_mapped.first);
        }
    }

    _frame_mtx.unlock();

    //We have all the cameras, we can now start a new thread in which the semantics will run.
    if(_external_semantics){
        boost::thread thread(&Segmenter::processFramesFromQueueExternal, this);
    }else{
        boost::thread thread(&Segmenter::processFramesFromQueueInternalRF, this);
    }
    boost::thread fusion_thread(&Segmenter::processMapFromQueue, this);

    //Create a projector.
    _projector = fps_mapper::MultiProjector();
    _projector.initFromCameraInfo(multi_image_node->cameraInfo());
    _projector.setImageSize(_cameras_in_order.size()*_camera_h, _camera_w);

    _projector.setMaxDistance(_depth_max);
    _projector.setMinDistance(_depth_min);
    _order_initialized = true;
    std::cout << "Cameras have been initialized and ordered correctly." << std::endl;
}

void Segmenter::onNewNode(fps_mapper::MapNode* m){
    //Check if it is a MultiImageMapNode pointer
    fps_mapper::MultiImageMapNode* m_multi = dynamic_cast<fps_mapper::MultiImageMapNode*>(m);
    if(m_multi){
        //Filter old messages
        if(_last_key_frame_id < m_multi->getId()){
            _last_key_frame_id = m_multi->getId();
            //Possibly compute the camera order here and setup a projector.
            if(!_order_initialized){
                initializeProjector(m_multi);
            }

            //Check if we want to use this key frame or we skip it.
            Eigen::Isometry3f pose = m_multi->transform();
            Eigen::Isometry3f delta = _last_pose.inverse()*pose;
            Eigen::AngleAxisf angle_delta(delta.linear());

            //Possibly skip it or use it as the last reference.
            if(delta.translation().norm() < _keyframe_skip_translation && fabs(angle_delta.angle()) < _keyframe_skip_rotation){
                std::cout << "skipping keyframe" << std::endl;
            }else{
                _frame_mtx.lock();

                _last_pose = pose;

                //We now know the order, get the frame ids.
                std::vector<int>& ids = m_multi->subimageSeqs();
                for(unsigned int i = 0; i < ids.size(); ++i){


                    std::pair< int, cv::Mat > color;
                    std::pair< int, cv::Mat > depth;
                    //Check if this is a request for an old frame, in such a case we cannot do anything really.
                    if(_cameras_in_order[i]->getIdAndClear(ids[i], color, depth)){
                        //The depth frame NEEDS to be there, if this was missed we'll have to drop it.
                        //Also skip storing the frame if the sequence number between the rgb and depth
                        //image diverges too much. For a small difference just use it as if it is a perfect match.
                        if(depth.first == ids[i] && depth.first - color.first < 3){
                            _image_queues[i].push_back(std::pair<int, std::pair<cv::Mat, cv::Mat> >(depth.first, std::pair<cv::Mat, cv::Mat>(depth.second, color.second)));
                            std::cout << i << " " << _image_queues[i].size() << std::endl;
                        }else{
                            std::cerr << "Dropped a frame for camera " << i << "! Semantics won't be computed for depth seq:" << ids[i] << std::endl;
                        }
                    }else{
                        //This shouldn't happen. If it does it means we just cannot perform a segmentation for that frame. It might already exist though!
                        //Debugging message. If this doesn't happen during testing it should be removed to make sure it works for the deployment.
                        throw std::runtime_error("Requested old missing frame for camera " + std::to_string(i) + "!");
                    }
                }
                _frame_mtx.unlock();
            }
        }
    }
}

void Segmenter::onNewLocalMap(fps_mapper::LocalMap* lmap) {
    _cloud_processing_mtx.lock();
    _local_map_queue.push_back(lmap);
    _cloud_processing_mtx.unlock();
}

//Call to actually start listening. 
void Segmenter::start(){
    //Init Giorgio's LocalMapListener part
    init(_n);

    // Subscribe to all cameras.
    for(std::map<std::string, Xtion*>::iterator it = _camera_map.begin(); it != _camera_map.end(); ++it){
        it->second->subscribeCamera(_it);
    }

    //Start the services that provide the semantic segmentation results.
    _local_map_ids_srv = _n.advertiseService("/semantic_segmentation/local_map_ids", &Segmenter::srvStoredSemanticsIds, this);
    _get_local_map_segmentation_srv = _n.advertiseService("/semantic_segmentation/get_local_map_segmentation", &Segmenter::srvGetLocalMapSegmentation, this);
    _semantic_information_srv = _n.advertiseService("/semantic_segmentation/information", &Segmenter::srvSegmentationInformation, this);
};

//Process the Single frames that are required using an external service.
void Segmenter::processFramesFromQueueInternalRF(){
    int max = 0;
    _frame_mtx.lock();
    max = _image_queues.size();
    _frame_mtx.unlock();


    std::vector<int> x_v, y_v;

    const unsigned int stride = _rf_prediction_stride;

    while(true){
        for(int i = 0; i < max; i++){
            _frame_mtx.lock();
            if(_image_queues[i].size()){
                //We have an image to segment! Let's send push it through the random forest

                std::pair<int, std::pair<cv::Mat, cv::Mat> >  ims =  _image_queues[i].front();
                _image_queues[i].pop_front();
                std:: cout << "got image" << std::endl;

                //Get the rgb and depth image.
                cv::Mat color = ims.second.second;
                cv::Mat depth = ims.second.first;
                Calibration& calib = _cameras_in_order[i]->getCalibration();

                x_v.clear();
                y_v.clear();
                libf::DataStorage storageT(_layer_count);
                _feature_extractor.extract(stride, color, depth, calib, storageT, x_v, y_v, ExtractType::NO_LABEL, _depth_min, _depth_max);
                std:: cout << "got features" << std::endl;
                //Init the result images
                std::vector<cv::Mat> result_ims(_layer_count);
                for(unsigned int layer = 0; layer < _layer_count; layer++){
                    result_ims[layer] = cv::Mat(color.rows/stride, color.cols/stride, CV_32FC(_layer_class_counts[layer]));
                    float* result_ptr = result_ims[layer].ptr<float>(0); // all this because OpenCV does not initialize above 4 channels..
                    for(int bla = 0; bla < result_ims[layer].cols * result_ims[layer].rows * result_ims[layer].channels(); bla++){
                        *result_ptr = 0;
                        result_ptr++;
                    }
                }

                //Get the RF output.
                for(int j = 0; j < storageT.getSize(); j++){
                    std::vector<std::vector<float> > post;
                    _random_forest.multiClassLogPosterior(storageT.getDataPoint(j), post);
                    for(unsigned int layer = 0; layer < _layer_count; layer++){
                        float* p = result_ims[layer].ptr<float>(y_v[j]/stride) + _layer_class_counts[layer]*x_v[j]/stride;
                        for(float pl : post[layer]){
                            *p = pl;
                            p++;
                        }
                    }
                }
                std:: cout << "got rs output" << std::endl;

                //Resize to full size
                for(unsigned int layer = 0; layer < _layer_count; layer++){
                    cv::resize(result_ims[layer], result_ims[layer], cv::Size(color.cols, color.rows));
                }
                /*
                //TMP
                for(unsigned int layer = 0; layer < _layer_count; layer++){
                    cv::Mat tmp_rgb_res(color.rows, color.cols, CV_8UC3);
                    for(int y = 0; y < color.rows; y++){
                        uchar* res_ptr = tmp_rgb_res.ptr<uchar>(y);
                        float* un_ptr = result_ims[layer].ptr<float>(y);
                        for(int x = 0; x < color.cols; x++){
                            uchar max = _layer_unknown_label[layer];
                            float max_val = -1000;
                            for(unsigned int l = 0; l < _layer_class_counts[layer]; l++){
                                float current_val = *un_ptr;
                                if (current_val > max_val){
                                        max_val = current_val;
                                        max = l;
                                }
                                un_ptr++;
                            }
                            res_ptr[0] = _layer_class_colors[layer][max][2];
                            res_ptr[1] = _layer_class_colors[layer][max][1];
                            res_ptr[2] = _layer_class_colors[layer][max][0];
                            res_ptr +=3;
                        }
                    }
                    Utils::ShowCvMat(tmp_rgb_res);
                }*/
                //Push them to a vector to be consistent with the service call.
                //TODO remove this if it is an issue!

                //Allocate memory (sum(all label counts) * width * height)
                unsigned int total_labels = 0;
                for(unsigned int layer = 0; layer < _layer_count; layer++){
                    total_labels += _layer_class_counts[layer];
                }
                std::vector<float> posteriors(total_labels*color.cols*color.rows, -1000.0);
                unsigned int index = 0;
                for(unsigned int layer = 0; layer < _layer_count; layer++){
                    unsigned int label_count = _layer_class_counts[layer];
                    for(int y = 0; y < color.rows; y++){
                        float* p = result_ims[layer].ptr<float>(y);
                        for(int x = 0; x < color.cols; x++){
                            for(unsigned int l = 0; l < label_count; l++){
                                posteriors[index] = *p;
                                p++;
                                index++;
                            }
                        }
                    }
                }

                //Store the result in the results queue
                _result_queues[i].push_back(std::pair<int, std::vector<float> >(ims.first, posteriors));
                _frame_mtx.unlock();
                std:: cout << "stored data" << std::endl;
            }else{
                _frame_mtx.unlock();
                std::this_thread::sleep_for( std::chrono::milliseconds(1));
            }
        }
    }
}

//Process the Single frames that are required using an external service.
void Segmenter::processFramesFromQueueExternal(){
    int max = 0;
    _frame_mtx.lock();
    max = _image_queues.size();
    _frame_mtx.unlock();

    while(true){
        for(int i = 0; i < max; i++){
            _frame_mtx.lock();
            if(_image_queues[i].size()){
                //We have an image to segment! Let's send it out to the service

                std::pair<int, std::pair<cv::Mat, cv::Mat> >  ims =  _image_queues[i].front();
                _image_queues[i].pop_front();

                //Get the rgb and depth image.
                cv::Mat rgb = ims.second.second;
                cv::Mat depth = ims.second.first;
                
                //Get the intrinsics, extrinsics and rectify the depth image.
                Eigen::MatrixXf mat(3, depth.rows*depth.cols);
                int index = 0;
                unsigned short* row_ptr = depth.ptr<unsigned short>();
                for(int y = 0; y < depth.rows; y++){
                    for(int x = 0; x < depth.cols; x++){
                        float depth = static_cast<float>(row_ptr[x])/1000.0f;
                        if( depth < 0.5 || depth > 15.0 ){
                            mat(0, index) = std::numeric_limits<float>::quiet_NaN();
                            mat(1, index) = std::numeric_limits<float>::quiet_NaN();
                            mat(2, index) = std::numeric_limits<float>::quiet_NaN();
                        }else{
                            mat(0, index) = depth * x;
                            mat(1, index) = depth * y;
                            mat(2, index) = depth;
                        }
                        index++;
                    }
                    row_ptr += depth.cols;
                }
                index = 0;
                Calibration& c = _cameras_in_order[i]->getCalibration();
                Eigen::MatrixXf rect = (c._extrinsic.linear()* c._intrinsic_inverse)*mat + c._extrinsic.translation().rowwise().replicate(depth.rows*depth.cols);
                cv::Mat depth3d(depth.rows, depth.cols,CV_32FC3,rect.data());

                //Call the segmentation service
                std_msgs::Header header;
                header.seq = ims.first;
                header.stamp = ros::Time::now();
                semantic_segmentation::SingleFrameSegmentation srv;
                cv_bridge::CvImage rgb_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, rgb);
                rgb_msg.toImageMsg(srv.request.rgb);
                cv_bridge::CvImage depth_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC3, depth3d);
                depth_msg.toImageMsg(srv.request.depth);
                //srv.request.
                if(_single_frame_segmentation_service.call(srv)){
                    //Store the result in the results queue
                    _result_queues[i].push_back(std::pair<int, std::vector<float> >(ims.first, srv.response.label_distribution));
                    std::cout << "stored images: " << _result_queues[i].size() << std::endl;
                }else{
                    throw std::runtime_error("Calling the segmentation service failed!");
                }
                _frame_mtx.unlock();
            }else{
                _frame_mtx.unlock();
                std::this_thread::sleep_for( std::chrono::milliseconds(1));
            }
        }
    }
}


//Fuses the single frame information into a Map.
void Segmenter::processMapFromQueue(){

    while(true){
        _cloud_processing_mtx.lock();
        if(_local_map_queue.size()){
            fps_mapper::LocalMap* lmap = _local_map_queue.front();

            //Check the ids of the last multi camera nodes and see if we have the semantics for each of them.

            //Get the last needed ids
            std::vector<int> last_ids;
            fps_mapper::MapNodeList& map_nodes = lmap->nodes();
            for(fps_mapper::MapNodeList::iterator m_it = map_nodes.begin(); m_it != map_nodes.end(); m_it++){

                fps_mapper::MultiImageMapNode* m_multi = dynamic_cast<fps_mapper::MultiImageMapNode*>((*m_it).get());
                if(m_multi){
                    last_ids = m_multi->subimageSeqs();
                }
            }
            //check if we can provide these.
            bool complete = true;
            _frame_mtx.lock();
            for(unsigned int i = 0; i < last_ids.size(); i++){
                if(_result_queues[i].back().first < last_ids[i]){
                    complete = false;
                    break;
                }
            }
            //If not:
            _frame_mtx.unlock();
            if(!complete){
                //We'll just postpone this for a little longer.
                _cloud_processing_mtx.unlock();
                std::this_thread::sleep_for( std::chrono::milliseconds(1));
                continue;
            }

            //Otherwise just continue as usual. Since we finish this local map now, pop it from the queue and go on.
            _local_map_queue.pop_front();
            _cloud_processing_mtx.unlock();
            //Get the cloud from the local map, this is needed to backproject points. 
            fps_mapper::Cloud* cloud = lmap->cloud();

            //allocate some memory for storing the probabilities
            unsigned int cloud_size = cloud->size();
            std::vector<Eigen::MatrixXf> unaries(_layer_count);
            for(unsigned int i = 0; i < _layer_count; i++){
                unsigned int label_count = _layer_class_counts[i];
                unaries[i] = Eigen::MatrixXf::Constant(label_count, cloud_size, 0.0);
            }

            int m = 0;
            for(fps_mapper::MapNodeList::iterator m_it = map_nodes.begin(); m_it != map_nodes.end(); m_it++){

                fps_mapper::MultiImageMapNode* m_multi = dynamic_cast<fps_mapper::MultiImageMapNode*>((*m_it).get());
                if(m_multi){
                    //get the index image.

                    FloatImage zbuffer;
                    IndexImage index_image; 
                    _projector.project(zbuffer, index_image, m_multi->transform().inverse(), *cloud);


                    std::vector<int>& subimage_seqs = m_multi->subimageSeqs();
                    for(unsigned int i = 0; i < subimage_seqs.size(); i++){

                        //For each of them, check if I have a segmentation.
                        _frame_mtx.lock();
                        std::vector<float> label_distribution;

                        //Drop skipped maps.
                        while(_result_queues[i].size() > 0 && _result_queues[i].front().first < subimage_seqs[i]){
                            _result_queues[i].pop_front();
                        }

                        //If this is the right id, we'll use it.
                        if(_result_queues[i].size() > 0 && _result_queues[i].front().first == subimage_seqs[i]){
                            label_distribution = _result_queues[i].front().second;
                            _result_queues[i].pop_front();
                            unsigned int offset = 0;
                            unsigned int current_class_count;
                            for(unsigned int l = 0; l < _layer_count; l++){
                                current_class_count = _layer_class_counts[l];
                                //Loop over the backprojected indices and project the semantic information into the cloud.
                                unsigned int image_index = 0;
                                for(unsigned int y = 0; y < _camera_h; y++){
                                    int* cloud_index_ptr = index_image.ptr<int>(y+i*_camera_h);
                                    for(unsigned int x = 0; x < _camera_w; x++){
                                        int index = cloud_index_ptr[x];
                                        if(index >= 0){
                                            for(unsigned int c = 0; c < current_class_count; c++){
                                                unaries[l](c,index) += label_distribution[offset + image_index*current_class_count + c];
                                            }
                                        }
                                        image_index++;
                                    }
                                }
                                offset += _camera_h * _camera_w * current_class_count;
                            }
                            std::cout << "found an image for " << subimage_seqs[i] << std::endl;
                        }else{
                            //We'll just ignore it. There might be missing patches in the final result.
                            std::cerr << "Couldn't find a semantic map for key frame: " << subimage_seqs[i] << std::endl;
                        }
                        _frame_mtx.unlock();
                    }
                    m++;
                }
            }
            std::vector<std::vector<unsigned char> > result_labels(_layer_count);
            if(_use_dense_crf){
                Eigen::MatrixXf pairwise(6, cloud_size);
                for(unsigned int i = 0 ; i < cloud_size; i++){
                    pairwise(0,i) = cloud->at(i)._point(0)*_dcrf_xyz_kernel;
                    pairwise(1,i) = cloud->at(i)._point(1)*_dcrf_xyz_kernel;
                    pairwise(2,i) = cloud->at(i)._point(2)*_dcrf_xyz_kernel;
                    pairwise(3,i) = cloud->at(i)._rgb(0)*_dcrf_rgb_kernel;
                    pairwise(4,i) = cloud->at(i)._rgb(1)*_dcrf_rgb_kernel;
                    pairwise(5,i) = cloud->at(i)._rgb(2)*_dcrf_rgb_kernel;
                }

                for(unsigned int l = 0; l < _layer_count; l++){
                    unsigned int class_count = _layer_class_counts[l];
                    DenseCRF crf(cloud_size, _layer_class_counts[l]);
                    crf.setUnaryEnergy(-unaries[l]);
                    crf.addPairwiseEnergy(pairwise, new PottsCompatibility(_dcrf_kernel_weight));
                    Eigen::MatrixXf res = crf.inference(_dcrf_iterations);
                    result_labels[l] = std::vector<unsigned char> (cloud_size);
                    for(unsigned int i = 0 ; i < cloud_size; i++){
                        unsigned int max = _layer_unknown_label[l];
                        float max_val = 2.0/class_count; //Only accept over a certain level of certainty?
                        for(unsigned int c = 0; c < class_count; c++){
                            float current = res(c, i);
                            if(current > max_val){
                                max_val = current;
                                max = c;
                            }
                        }
                        result_labels[l][i] = max;
                    }
                }
            }else{
                for(unsigned int l = 0; l < _layer_count; l++){
                    unsigned int class_count = _layer_class_counts[l];
                    result_labels[l] = std::vector<unsigned char> (cloud_size);
                    for(unsigned int i = 0 ; i < cloud_size; i++){
                        unsigned int max = _layer_unknown_label[l];
                        float max_val = -1000; //TODO init with unknown label here?
                        float sum = 0;
                        for(unsigned int c = 0; c < class_count; c++){
                            float current = unaries[l](c, i);
                            sum += current;
                            if(current > max_val){
                                max_val = current;
                                max = c;
                            }
                        }
                        if(sum != 0.0){
                            result_labels[l][i] = max;
                        }else{
                            result_labels[l][i] = _layer_unknown_label[l];
                        }
                    }
                }
            }
            //For debugging and videos, let's just dump everything to /tmp, yes this is fixed :D
            if(_dump_clouds_to_tmp){

                //Color for comparison
                std::ofstream os("/tmp/cloud" + std::to_string(lmap->getId())+ "_rgb.cld");
                cloud->write(os);

                for(unsigned int layer = 0; layer < _layer_count; layer++){
                    for(unsigned int i = 0 ; i < cloud_size; i++){
                        // For "nicer" visualization of points without normals.
                        cloud->at(i).normalize();
                        if(cloud->at(i).normal().squaredNorm() < 0.1){
                            cloud->at(i)._normal = Eigen::Vector3f(0,0,1);
                        }
                        unsigned char current_label = result_labels[layer][i];
                        cloud->at(i)._rgb(0) = static_cast<float>(_layer_class_colors[layer][current_label][0])/255.0;
                        cloud->at(i)._rgb(1) = static_cast<float>(_layer_class_colors[layer][current_label][1])/255.0;
                        cloud->at(i)._rgb(2) = static_cast<float>(_layer_class_colors[layer][current_label][2])/255.0;
                    }
                    //TMP
                    std::ofstream os_labeled("/tmp/cloud" + std::to_string(lmap->getId())+ "_layer_" + std::to_string(layer) + ".cld");
                    cloud->write(os_labeled);
                }
            }

            // TODO : clean up old images from the map before. Assumption: if it wasn't used in this map anymore, it is safe to go?

            //Save data for the service based on the map id.
            _cloud_mtx.lock();
            _cloud_results.push_back(std::pair<int, std::vector<std::vector<unsigned char> > >(lmap->getId(),result_labels));
            _cloud_mtx.unlock();
        }else{
            _cloud_processing_mtx.unlock();
        }
        std::this_thread::sleep_for( std::chrono::milliseconds(1));
    }
}


bool Segmenter::srvStoredSemanticsIds(semantic_segmentation::IdsSrv::Request& req, semantic_segmentation::IdsSrv::Response& resp){
    _cloud_mtx.lock();
    for(auto m : _cloud_results){
        resp.local_map_ids.push_back(m.first);
    }
    _cloud_mtx.unlock();
    return true;
}

bool Segmenter::srvGetLocalMapSegmentation(semantic_segmentation::LocalMapSegmentationSrv::Request& req, semantic_segmentation::LocalMapSegmentationSrv::Response& resp){
    //Get indices for the names of the requested layers.
    std::vector<int> layer_indices;
    for(std::string l :  req.segmentation_layers){
        for(unsigned int i = 0; i < _layer_count; i++){
            if(l.compare(_layer_names[i]) == 0){
                layer_indices.push_back(i);
                break;
            }
        }
    }

    //Check if we got the right amount of layer indices.
    if(req.segmentation_layers.size() != layer_indices.size()){
        return false;
    }

    _cloud_mtx.lock();
    for(auto m : _cloud_results){
        if(m.first == req.local_map_id){
            std::vector<std::vector<unsigned char> >& result_labels = m.second;
            resp.local_map_id = m.first;
            _cloud_mtx.unlock();

            unsigned int point_count = result_labels[0].size();

            //Reserve space
            resp.point_labels.reserve(point_count * layer_indices.size());

            for(int l : layer_indices){
                for(unsigned int i = 0; i < point_count; i++){
                    int current = result_labels[l][i];
                    if(current >= _layer_class_names[l].size() || current < 0){
                        std::cout << "whut " << current << " " << l <<  " "  << _layer_class_names[l].size() << std::endl;
                    }
                    resp.point_labels.push_back(result_labels[l][i]);
                }
            }
            return true;
        }
    }
    _cloud_mtx.unlock();
    return false;
}

bool Segmenter::srvSegmentationInformation(semantic_segmentation::SegmentationInformationSrv::Request& req, semantic_segmentation::SegmentationInformationSrv::Response& resp){
    resp.layer_names = _layer_names;
    resp.class_counts = _layer_class_counts;
    for(std::vector<std::string> l_c : _layer_class_names){
        for(std::string c : l_c){
            resp.class_names.push_back(c);
        }
    }
    for(std::vector<std::vector<unsigned char> > l_col : _layer_class_colors){
        for(std::vector<unsigned char> c : l_col){
            resp.class_colors.push_back(c[0]);
            resp.class_colors.push_back(c[1]);
            resp.class_colors.push_back(c[2]);
        }
    }
    return true;
}