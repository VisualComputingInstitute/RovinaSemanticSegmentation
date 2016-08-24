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

    //Check if the config suggests we should load a multi label rf. in that case stop as we should use the other executable.
    std::string label_type_prefix = conf.get<std::string>("training_label_prefix");
    if(label_type_prefix.compare("shared") ==0){
      throw std::runtime_error("The config file wants to load a mutli label model. Please use the test_multi executable.");
    }

    //Parse the color codings.
    std::map<std::string, std::string> color_codings;
    Json::Value coding_list = conf.getRaw("color_codings");
    for(auto l :coding_list){
      std::string name = l["name"].asString();
      std::string conv = l["coding"].toStyledString();
      color_codings[name] = conv;
    }

    if(color_codings.count(label_type_prefix) == 0){
      throw std::runtime_error("The required color coding was missing. (" + label_type_prefix + ")");
    }
    RgbLabelConversion label_conv(color_codings[label_type_prefix]);

    //Load calibration
    std::string calibration_dir = conf.getPath("calibration_dir");
    std::string calibration_ext = conf.get<std::string>("calibration_ext");

    //Get a list of image names.
    std::vector<std::string> filenames_test = conf.getFromFile<std::vector<std::string>>("file_names_test");
    std::string color_dir = conf.getPath("color_dir");
    std::string color_ext = conf.get<std::string>("color_ext");

    std::string depth_dir = conf.getPath("depth_dir");
    std::string depth_ext = conf.get<std::string>("depth_ext");


    std::string label_dir = conf.getPath(label_type_prefix+"_label_dir");
    std::string label_ext = conf.get<std::string>(label_type_prefix+"_label_ext");


    std::string result_dir = conf.getPath(label_type_prefix+"_result_dir");
    std::string result_ext = conf.get<std::string>(label_type_prefix+"_result_ext");

    std::string model_file = conf.getPath(label_type_prefix + "_forest_file_name");

    float dmin = conf.get<float>("depth_min");
    float dmax = conf.get<float>("depth_max");

    libf::DataStorage storage;

    std::vector<int> x_v, y_v;

    Features::FeatureExtractor f(conf);

    libf::RandomForest* forest = new libf::RandomForest();

    std::filebuf fb;
    if (fb.open (model_file,std::ios::in)){
      std::istream is(&fb);
      forest->read(is);
    }
    fb.close();

    std::cout << "starting to extract ... " << std::endl;

    unsigned int stride = conf.get<unsigned int>("rf_prediction_stride");

    int labels = label_conv.getValidLabelCount();
    std::vector<int> label_count(labels*labels,0);
    std::vector<int> class_count(labels,0);
    std::vector<int> vote_count(labels,0);

    int total = 0;
    float time_avg = 0;
    int img_count = 0;
    for(unsigned int i = 0; i < filenames_test.size(); ++i){
      cv::Mat color = cv::imread(color_dir + filenames_test[i] + color_ext, CV_LOAD_IMAGE_COLOR);
      cv::cvtColor(color, color, CV_BGR2RGB); //As the trained model expects this.
      cv::Mat depth_im = cv::imread(depth_dir + filenames_test[i] + depth_ext, CV_LOAD_IMAGE_ANYDEPTH);
      std::vector<cv::Mat> label;
      label.push_back(label_conv.rgbToLabel(cv::imread(label_dir + filenames_test[i] + label_ext, CV_LOAD_IMAGE_COLOR)));
      Calibration c(calibration_dir + filenames_test[i] +  calibration_ext);
      std::cout << "loading done" << std::endl;
      x_v.clear();
      y_v.clear();
      libf::DataStorage storageT;
      const clock_t begin_time = clock();
      f.extract(stride, color, depth_im, c, storageT, x_v, y_v, ExtractType::WITH_ANY_LABEL, dmin, dmax, label);
      cv::Mat result_im(color.rows/2, color.cols/2, CV_32FC(labels));
      std::cout << "extraction done" << std::endl;
      float* result_ptr = result_im.ptr<float>(0); // all this because OpenCV does not initialize above 4 channels..
      for(int bla = 0; bla < result_im.cols * result_im.rows * result_im.channels(); bla++){
        *result_ptr = -1000;
        result_ptr++;
      }

      for(int j = 0; j < storageT.getSize(); j++){
        std::vector<float> post;
        forest->classLogPosterior(storageT.getDataPoint(j), post);
        float* p = result_im.ptr<float>(y_v[j]/2) + labels*x_v[j]/2;
        for(float pl : post){
         *p = pl;
         p++;
        }
      }
      cv::resize(result_im, result_im, cv::Size(color.cols, color.rows));
      cv::Mat result_im_char(color.rows, color.cols, CV_8SC1);
      for(int y = 0; y < color.rows; y++){
        float* p = result_im.ptr<float>(y);
        char* r = result_im_char.ptr<char>(y);
        for(int x = 0; x < color.cols; x++){
          *r = -1;
          float max = -1000;
          for(int l = 0; l < labels; l++){
            if(*p > max){
              max = *p;
              *r = l;
            }
            p++;
          }
          r++;
        }
      }

      time_avg += float( clock () - begin_time ) /  CLOCKS_PER_SEC;
      img_count++;
      std::cout << "x" << std::flush;

//       cv::imshow("result", object_label_conv.labelToRgb(result_im));
//       cv::imshow("gt", object_label_conv.labelToRgb(label));
//       cv::imshow("rgb",color_org);
//       cv::waitKey(0);
      char* res = result_im_char.ptr<char>(0);
      char* gt = label[0].ptr<char>(0);

      for(int x = 0; x < label[0].rows*label[0].cols; x++){
        if(*res >= 0 && *gt >= 0){
          label_count[(*gt)*labels + *res] ++;
          class_count[*gt]++;
          vote_count[*res]++;
          total++;
        }
        res++;
        gt++;
      }

      cv::imwrite(result_dir + filenames_test[i] + result_ext, label_conv.labelToRgb(result_im_char));
    }
    std::cout << std::endl << "Time per image: " << time_avg/img_count << std::endl;

    int total_acc = 0;
    float avg_acc = 0;
    float iou = 0;
    std::cout << "confusion:" << std::endl;
    int l = 0;
    for(int i = 0; i < labels; i++){
      std::string n = label_conv.getLabelName(i);
      for(int p = n.length(); p < 15; p++){
        n+= " ";
      }
      std::cout << n;
      for(int j = 0; j < labels; j++){
        if( i == j){
          total_acc+=label_count[l];
          avg_acc += 100.0*static_cast<float>(label_count[l]) / (class_count[i] ? class_count[i] : 1);
          int x = class_count[i] + vote_count[i] - label_count[l];
          iou += 100.0*static_cast<float>(label_count[l]) / ( x ? x : 1);
        }
        printf(" %6.2f", 100.0*static_cast<float>(label_count[l]) / (class_count[i] ? class_count[i] : 1));
        l++;
      }
      std::cout << "   out of " << class_count[i] << " pixels" <<  std::endl;
    }
    printf("Global accuracy:         %6.2f \n", 100.0*static_cast<float>(total_acc)/total);
    printf("Class averge accuracy:   %6.2f \n", avg_acc/labels);
    printf("Intersection over union: %6.2f \n", iou/labels);

    return 0;
}
