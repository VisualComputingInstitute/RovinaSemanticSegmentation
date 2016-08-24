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

    //Check if the config suggests we should load a normal rf. in that case stop as we should use the other executable.
    std::string label_type_prefix = conf.get<std::string>("training_label_prefix");
    if(label_type_prefix.compare("shared") !=0){
      throw std::runtime_error("The config file wants to load a material/object model. Please use the normal test executable.");
    }

    //For now we assume two fixed layers, change stuff up here and it should be automatic in the lower parts of the code.
    //Not tested obviously :D If you ever do this, make sure you train the RF with the same order as you push back stuff here.
    const unsigned int layer_count = 2;

    const unsigned int stride = conf.get<unsigned int>("rf_prediction_stride");;

    //Parse the color codings.
    std::map<std::string, std::string> color_codings;
    Json::Value coding_list = conf.getRaw("color_codings");
    for(auto l :coding_list){
      std::string name = l["name"].asString();
      std::string conv = l["coding"].toStyledString();
      color_codings[name] = conv;
    }

    if(color_codings.count("material") == 0 || color_codings.count("object") == 0){
      throw std::runtime_error("One of the required color codings (material,object) was missing.");
    }
    std::vector<RgbLabelConversion> label_conv;
    label_conv.push_back(RgbLabelConversion(color_codings["material"]));
    label_conv.push_back(RgbLabelConversion(color_codings["object"]));

    std::vector<std::string> label_dir;
    label_dir.push_back(conf.getPath("material_label_dir"));
    label_dir.push_back(conf.getPath("object_label_dir"));

    std::vector<std::string> label_ext;
    label_ext.push_back(conf.get<std::string>("material_label_ext"));
    label_ext.push_back(conf.get<std::string>("object_label_ext"));

    std::vector<std::string> result_dir;
    result_dir.push_back(conf.getPath("material_result_dir"));
    result_dir.push_back(conf.getPath("object_result_dir"));

    std::vector<std::string> result_ext;
    result_ext.push_back(conf.get<std::string>("material_result_ext"));
    result_ext.push_back(conf.get<std::string>("object_result_ext"));

    std::string model_file = conf.getPath("forest_file_name");

    float dmin = conf.get<float>("depth_min");
    float dmax = conf.get<float>("depth_max");

    //Load calibration
    std::string calibration_dir = conf.getPath("calibration_dir");
    std::string calibration_ext = conf.get<std::string>("calibration_ext");

    //Get a list of image names.
    std::vector<std::string> filenames_test = conf.getFromFile<std::vector<std::string>>("file_names_test");
    std::string color_dir = conf.getPath("color_dir");
    std::string color_ext = conf.get<std::string>("color_ext");

    std::string depth_dir = conf.getPath("depth_dir");
    std::string depth_ext = conf.get<std::string>("depth_ext");

    std::cout << "starting to extract ... " << std::endl;

    libf::DataStorage storage(layer_count);

    std::vector<int> x_v, y_v;

    Features::FeatureExtractor f(conf);

    libf::RandomForest* forest = new libf::RandomForest();

    std::filebuf fb;
    if (fb.open (model_file,std::ios::in)){
      std::istream is(&fb);
      forest->read(is);
    }
    fb.close();

    std::vector<int> layer_labelcounts(layer_count);
    std::vector<std::vector<int> > label_counts(layer_count);
    std::vector<std::vector<int> > class_counts(layer_count);
    std::vector<std::vector<int> > vote_counts(layer_count);
    std::vector<int> totals(layer_count,0);
    
    for(unsigned int layer = 0; layer < layer_count; layer++){
      unsigned int layer_label_count = label_conv[layer].getValidLabelCount();
      std::cout << layer_label_count << std::endl;
      layer_labelcounts[layer] = layer_label_count;
      label_counts[layer] = std::vector<int> (layer_label_count*layer_label_count,0);
      class_counts[layer] = std::vector<int> (layer_label_count,0);
      vote_counts[layer] = std::vector<int> (layer_label_count,0);
    }

    //For timing
    float time_avg = 0;
    int img_count = 0;


    for(unsigned int i = 0; i < filenames_test.size(); ++i){
      std::cout << "+" << std::flush;
      cv::Mat color = cv::imread(color_dir + filenames_test[i] + color_ext, CV_LOAD_IMAGE_COLOR);
      cv::cvtColor(color, color, CV_BGR2RGB); //As the trained model expects this.
      cv::Mat depth_im = cv::imread(depth_dir + filenames_test[i] + depth_ext, CV_LOAD_IMAGE_ANYDEPTH);

      std::vector<cv::Mat> labels(layer_count);
      for(unsigned int layer = 0; layer < layer_count; layer++){
        labels[layer] = label_conv[layer].rgbToLabel(cv::imread(label_dir[layer] + filenames_test[i] + label_ext[layer], CV_LOAD_IMAGE_COLOR));
      }

      Calibration c(calibration_dir + filenames_test[i] +  calibration_ext);
      x_v.clear();
      y_v.clear();

      libf::DataStorage storageT(layer_count);
      const clock_t begin_time = clock();
      f.extract(stride, color, depth_im, c, storageT, x_v, y_v, ExtractType::WITH_ANY_LABEL, dmin, dmax, labels);

      //Init the result images
      std::vector<cv::Mat> result_ims(layer_count);
      for(unsigned int layer = 0; layer < layer_count; layer++){

        result_ims[layer] = cv::Mat(color.rows/stride, color.cols/stride, CV_32FC(layer_labelcounts[layer]));

        float* result_ptr = result_ims[layer].ptr<float>(0); // all this because OpenCV does not initialize above 4 channels..
        for(int bla = 0; bla < result_ims[layer].cols * result_ims[layer].rows * result_ims[layer].channels(); bla++){
          *result_ptr = -1000;
          result_ptr++;
        }
      }

      //Get the RF output.
      for(int j = 0; j < storageT.getSize(); j++){
        std::vector<std::vector<float> > post;
        forest->multiClassLogPosterior(storageT.getDataPoint(j), post);
        for(unsigned int layer = 0; layer < layer_count; layer++){
          float* p = result_ims[layer].ptr<float>(y_v[j]/stride) + layer_labelcounts[layer]*x_v[j]/stride;
          for(float pl : post[layer]){
            *p = pl;
            p++;
          }
        }
      }
      for(unsigned int layer = 0; layer < layer_count; layer++){
        cv::resize(result_ims[layer], result_ims[layer], cv::Size(color.cols, color.rows));
        cv::Mat result_im_char(color.rows, color.cols, CV_8SC1);
        unsigned int label_count = layer_labelcounts[layer];
        for(int y = 0; y < color.rows; y++){
          float* p = result_ims[layer].ptr<float>(y);
          char* r = result_im_char.ptr<char>(y);
          for(int x = 0; x < color.cols; x++){
            *r = -1;
            float max = -1000;
            for(unsigned int l = 0; l < label_count; l++){
              if(*p > max){
                max = *p;
                *r = l;
              }
              p++;
            }
            r++;
          }
        }
        cv::imwrite(result_dir[layer] + filenames_test[i] + result_ext[layer], label_conv[layer].labelToRgb(result_im_char));
        char* res = result_im_char.ptr<char>(0);
        char* gt = labels[layer].ptr<char>(0);

        for(int x = 0; x < color.rows*color.cols; x++){
          if(*res >= 0 && *gt >= 0){
            label_counts[layer][(*gt)*label_count + *res] ++;
            class_counts[layer][*gt]++;
            vote_counts[layer][*res]++;
            totals[layer]++;
          }
          res++;
          gt++;
        }
      }

      time_avg += float( clock () - begin_time ) /  CLOCKS_PER_SEC;
      img_count++;
      std::cout << "x" << std::flush;
    }
    std::cout << std::endl << "Time per image: " << time_avg/img_count << std::endl;

    for(unsigned int layer = 0; layer < layer_count; layer++){
      unsigned int label_count = layer_labelcounts[layer];
      int total_acc = 0;
      float avg_acc = 0;
      float iou = 0;
      std::cout << "confusion:" << std::endl;
      int l = 0;
      for(unsigned int i = 0; i < label_count; i++){
        std::string n = label_conv[layer].getLabelName(i);
        for(int p = n.length(); p < 15; p++){
          n+= " ";
        }
        std::cout << n;
        for(unsigned int j = 0; j < label_count; j++){
          if( i == j){
            total_acc+=label_counts[layer][l];
            avg_acc += 100.0*static_cast<float>(label_counts[layer][l]) / (class_counts[layer][i] ? class_counts[layer][i] : 1);
            int x = class_counts[layer][i] + vote_counts[layer][i] - label_counts[layer][l];
            iou += 100.0*static_cast<float>(label_counts[layer][l]) / ( x ? x : 1);
          }
          printf(" %6.2f", 100.0*static_cast<float>(label_counts[layer][l]) / (class_counts[layer][i] ? class_counts[layer][i] : 1));
          l++;
        }
        std::cout << "   out of " << class_counts[layer][i] << " pixels" <<  std::endl;
      }
      printf("Global accuracy:         %6.2f \n", 100.0*static_cast<float>(total_acc)/totals[layer]);
      printf("Class averge accuracy:   %6.2f \n", avg_acc/label_count);
      printf("Intersection over union: %6.2f \n", iou/label_count);
    }
    return 0;
}
