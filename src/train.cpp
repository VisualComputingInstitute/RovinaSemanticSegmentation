//STL includes
#include <ctime>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

//OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


// local includes
#include "feature_extractor.h"
#include "json/json.h"
#include "calibration.h"
#include "config.h"
#include "commandline_parser.h"
#include "cv_util.h"
#include "defines.h"
#include "rgb_label_conversion.h"

//libforest
#include "libforest/data.h"
#include "libforest/libforest.h"

std::string usage(char * exe){
  std::string u("\n");
  u += std::string(exe) + " --conf <config file> --opt1 <val1> ... \n";
  return u;
}

int main(int argc, char **argv) {
    if(argc <= 2){
      throw std::runtime_error("No parameters given. Usage: " + usage(argv[0]));
    }

    //Parse all parameters
    std::map<std::string, std::string> parameter_name_value_map;
    bool parse_succeeded = Utils::parseParamters(argc, argv, parameter_name_value_map);
    if(!parse_succeeded){
      throw std::runtime_error("Mangled command line arguments. " + usage(argv[0]));
    }

    //check if we found a config file.
    if(parameter_name_value_map.count("conf") == 0){
      throw std::runtime_error("No config file was given" + usage(argv[0]));
    }

    std::string config_file = parameter_name_value_map["conf"];
    parameter_name_value_map.erase("conf");
    Utils::Config conf(config_file, parameter_name_value_map);


    //Load calibration
    std::string calibration_dir = conf.getPath("calibration_dir");
    std::string calibration_ext = conf.get<std::string>("calibration_ext");

    //Get a list of image names.
    std::vector<std::string> filenames_train = conf.getFromFile<std::vector<std::string>>("file_names_train");
    std::string color_dir = conf.getPath("color_dir");
    std::string color_ext = conf.get<std::string>("color_ext");

    std::string depth_dir = conf.getPath("depth_dir");
    std::string depth_ext = conf.get<std::string>("depth_ext");

    std::string label_type_prefix = conf.get<std::string>("training_label_prefix");

    std::string model_file;


    libf::DataStorage storage;


    //Parse the color codings. Somewhat ugly but needed to converge the two ways color codings are handled atm. :(
    std::map<std::string, std::string> color_codings;
    Json::Value coding_list = conf.getRaw("color_codings");
    for(auto l :coding_list){
      std::string name = l["name"].asString();
      std::string conv = l["coding"].toStyledString();
      color_codings[name] = conv;
    }

    float dmin = conf.get<float>("depth_min");
    float dmax = conf.get<float>("depth_max");

    if(label_type_prefix.compare("shared") ==0){

      if(color_codings.count("material") == 0 || color_codings.count("object") == 0){
        throw std::runtime_error("One of the required color codings (material,object) was missing.");
      }
      RgbLabelConversion mat_label_conv(color_codings["material"]);
      RgbLabelConversion obj_label_conv(color_codings["object"]);
      
      std::string mat_label_dir = conf.getPath("material_label_dir");
      std::string mat_label_ext = conf.get<std::string>("material_label_ext");

      std::string obj_label_dir = conf.getPath("object_label_dir");
      std::string obj_label_ext = conf.get<std::string>("object_label_ext");

      model_file = conf.getPath("forest_file_name");

      std::cout << "starting to extract ... " << std::endl;

      int stride = conf.get<int>("training_sample_stride");

      storage = libf::DataStorage(2);

      std::vector<int> x_v, y_v;

      Features::FeatureExtractor f(conf);

      std::vector<char> augment(3,0);
      augment[0] = -20;
      augment[2] = 20;


      for(unsigned int i = 0; i < filenames_train.size(); ++i){
        for(auto a : augment){
          cv::Mat color = cv::imread(color_dir + filenames_train[i] + color_ext, CV_LOAD_IMAGE_COLOR);
          cv::cvtColor(color, color, CV_BGR2RGB); //So the trained model can also be used on direct xtion data.
          color += a;
          cv::Mat depth_im = cv::imread(depth_dir + filenames_train[i] + depth_ext, CV_LOAD_IMAGE_ANYDEPTH);
          std::vector<cv::Mat> labels;
          labels.push_back(mat_label_conv.rgbToLabel(cv::imread(mat_label_dir + filenames_train[i] + mat_label_ext, CV_LOAD_IMAGE_COLOR)));
          labels.push_back(obj_label_conv.rgbToLabel(cv::imread(obj_label_dir + filenames_train[i] + obj_label_ext, CV_LOAD_IMAGE_COLOR)));
          Calibration c(calibration_dir + filenames_train[i] +  calibration_ext);
          std::cout << "x" << std::flush;

          //Flip it for super easy augmentation
          f.extract(stride, color, depth_im, c, storage, x_v, y_v, ExtractType::WITH_POSITIVE_LABEL, dmin, dmax, labels);
          cv::Mat color_f;
          cv::flip(color, color_f,1);
          cv::Mat depth_im_f;
          cv::flip(depth_im, depth_im_f,1);
          cv::Mat mat_label_f;
          cv::flip(labels[0], mat_label_f,1);
          cv::Mat obj_label_f;
          cv::flip(labels[1], obj_label_f,1);
          std::vector<cv::Mat> labels_f;
          labels_f.push_back(mat_label_f);
          labels_f.push_back(obj_label_f);
          f.extract(stride, color_f, depth_im_f, c, storage, x_v, y_v, ExtractType::WITH_POSITIVE_LABEL, dmin, dmax, labels_f);
          std::cout << "+" << std::flush;
        }
      }
      std::cout << storage.getSize() << std::endl;
      for(int ml = 0; ml < storage.getMultiLayerCount(); ml ++){
        std::map <int, int> dist;
        for(int i = 0; i < storage.getSize(); i++){
          int l = storage.getClassLabelsMulti(i, ml);
          if(dist.count(l)){
            dist[l]++;
          }else{
            dist[l] = 1;
          }
        }

        for(auto label_freq : dist){
          std::cout << label_freq.first << "->" << label_freq.second << std::endl;
        }
      }
    }else{

      //Parse the color codings.
      if(color_codings.count(label_type_prefix) == 0){
        throw std::runtime_error("The required color coding was missing. (" + label_type_prefix + ")");
      }
      RgbLabelConversion label_conv(color_codings[label_type_prefix]);

      std::string label_dir = conf.getPath(label_type_prefix+"_label_dir");
      std::string label_ext = conf.get<std::string>(label_type_prefix+"_label_ext");

      model_file = conf.getPath(label_type_prefix+"_forest_file_name");

      std::cout << "starting to extract ... " << std::endl;

      int stride = conf.get<int>("training_sample_stride");

      std::vector<int> x_v, y_v;

      Features::FeatureExtractor f(conf);


      for(unsigned int i = 0; i < filenames_train.size(); ++i){
        cv::Mat color = cv::imread(color_dir + filenames_train[i] + color_ext, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(color, color, CV_BGR2RGB); //So the trained model can also be used on direct xtion data.
        cv::Mat depth_im = cv::imread(depth_dir + filenames_train[i] + depth_ext, CV_LOAD_IMAGE_ANYDEPTH);
        std::vector<cv::Mat> labels;
        labels.push_back(label_conv.rgbToLabel(cv::imread(label_dir + filenames_train[i] + label_ext, CV_LOAD_IMAGE_COLOR)));
        Calibration c(calibration_dir + filenames_train[i] +  calibration_ext);
        std::cout << "x" << std::flush;

        //Flip it for super easy augmentation
        f.extract(stride, color, depth_im, c, storage, x_v, y_v, ExtractType::WITH_POSITIVE_LABEL, dmin, dmax, labels);
        cv::Mat color_f;
        cv::flip(color, color_f,1);
        cv::Mat depth_im_f;
        cv::flip(depth_im, depth_im_f,1);
        cv::Mat label_f;
        cv::flip(labels[0], label_f,1);
        std::vector<cv::Mat> labels_f;
        labels_f.push_back(label_f);
        f.extract(stride, color_f, depth_im_f, c, storage, x_v, y_v, ExtractType::WITH_POSITIVE_LABEL, dmin, dmax, labels_f);
        std::cout << "+" << std::flush;
      }
      std::cout << storage.getSize() << std::endl;
      std::map <int, int> dist;
      for(int i = 0; i < storage.getSize(); i++){
        int l = storage.getClassLabel(i);
        if(dist.count(l)){
          dist[l]++;
        }else{
          dist[l] = 1;
        }
      }

      for(auto label_freq : dist){
        std::cout << label_freq.first << "->" << label_freq.second << std::endl;
      }
    }

    libf::DecisionTreeLearner treeLearner;


    treeLearner.autoconf(&storage);
    treeLearner.setUseBootstrap(true);
    treeLearner.setMaxDepth(conf.get<int>("max_depth"));
    treeLearner.setMinSplitExamples(conf.get<int>("min_split_sample"));
    treeLearner.setUseClassFrequency(false);
    treeLearner.useMultiLabelLayers(label_type_prefix.compare("shared") == 0);
    libf::RandomForestLearner forestLearner;
    forestLearner.addCallback(libf::RandomForestLearner::defaultCallback, 1);

    forestLearner.setTreeLearner(&treeLearner);
    forestLearner.setNumTrees(conf.get<int>("num_trees"));
    forestLearner.setNumThreads(8);

    libf::RandomForest* forest = forestLearner.learn(&storage);


    std::filebuf fb;
    if (fb.open (model_file,std::ios::out)){
      std::ostream os(&fb);
      forest->write(os);
    }
    fb.close();



    return 0;
}
