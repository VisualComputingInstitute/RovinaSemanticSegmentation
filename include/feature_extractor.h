#pragma once

#include "calibration.h"
#include "defines.h"

// Local includes
#include "config.h"

//libforest includes
#include "libforest/libforest.h"

//OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

//PCL includes
#include <pcl/features/integral_image_normal.h>


enum class ExtractType { WITH_ANY_LABEL, WITH_POSITIVE_LABEL, NO_LABEL};

namespace Features{

class FeatureExtractor {
public:
  FeatureExtractor(){};

  FeatureExtractor(Utils::Config conf){
    _feature_usage.push_back(conf.get<bool>("feature_color_patch"));
    _feature_usage.push_back(conf.get<bool>("feature_depth"));
    _feature_usage.push_back(conf.get<bool>("feature_height"));
    _feature_usage.push_back(conf.get<bool>("feature_normal"));

    _patch_size = conf.get<int>("patch_size");  //31
    _patch_size_reduce = conf.get<int>("patch_size_reduce");  //5
    _border = _patch_size; // assume nothing is closer than 0.5cm to the camera.  (border = patch_size /2 * 2)

  }

  void extract(int stride, cv::Mat& color, cv::Mat& depth_im, Calibration c, libf::DataStorage& storage, std::vector<int>& x_v, std::vector<int>& y_v, ExtractType label_extraction, float d_min, float d_max, std::vector<cv::Mat> label = std::vector<cv::Mat>()){

    const float d_min_mm= d_min*1000.0;
    const float d_max_mm= d_max*1000.0;

    //First compute the feature length.
    int feature_length = 0;
    if(_feature_usage[0]) feature_length += _patch_size_reduce*_patch_size_reduce*3; //color patch 3 channels.
    if(_feature_usage[1]) feature_length += 1; // depth3
    if(_feature_usage[2]) feature_length += 1; // height
    if(_feature_usage[3]) feature_length += 1; // normal


    int original_datastorage_size = storage.getSize();
    //Create the datapoints
    cv::Mat mask(depth_im.rows, depth_im.cols, CV_8UC1, cv::Scalar(0));
    if(label_extraction == ExtractType::NO_LABEL){
      for(int y = 0; y < depth_im.rows; y+=stride){
        unsigned short* depth_ptr = depth_im.ptr<unsigned short>(y);
        unsigned char* mask_ptr = mask.ptr<unsigned char>(y);
        for(int x = 0; x < depth_im.cols; x+=stride){
          if(*depth_ptr >= d_min_mm && *depth_ptr <= d_max_mm){
            storage.addDataPoint(new libf::DataPoint(feature_length), -1);
            x_v.push_back(x);
            y_v.push_back(y);
            *mask_ptr = 1;
          }
          mask_ptr+=stride;
          depth_ptr+=stride;
        }
      }
    }else{
      if(label.size() == 1){
        for(int y = 0; y < depth_im.rows; y+=stride){
          unsigned short* depth_ptr = depth_im.ptr<unsigned short>(y);
          label_type* label_ptr = label[0].ptr<label_type>(y);
          unsigned char* mask_ptr = mask.ptr<unsigned char>(y);
          for(int x = 0; x < depth_im.cols; x+=stride){
            if(*depth_ptr >= d_min_mm && *depth_ptr <= d_max_mm && (label_extraction == ExtractType::WITH_ANY_LABEL || *label_ptr >= 0)){
              storage.addDataPoint(new libf::DataPoint(feature_length), *label_ptr);
              x_v.push_back(x);
              y_v.push_back(y);
              *mask_ptr = 1;
            }
            mask_ptr+=stride;
            depth_ptr+=stride;
            label_ptr+=stride;
          }
        }
      }else{
        for(int y = 0; y < depth_im.rows; y+=stride){
          unsigned short* depth_ptr = depth_im.ptr<unsigned short>(y);
          std::vector<label_type*> label_ptr;
          for(unsigned int l = 0; l < label.size(); l++){
            label_ptr.push_back(label[l].ptr<label_type>(y));
          }
          unsigned char* mask_ptr = mask.ptr<unsigned char>(y);
          for(int x = 0; x < depth_im.cols; x+=stride){
            bool labels_okay = true;
            for(label_type* l_ptr : label_ptr){
              labels_okay &= (*l_ptr >=0);
            }
            if(*depth_ptr >= d_min_mm && *depth_ptr <= d_max_mm && (label_extraction == ExtractType::WITH_ANY_LABEL || labels_okay)){
              std::vector<int> lab;
              for(label_type* l_ptr : label_ptr){
                lab.push_back(*l_ptr);
              }
              storage.addDataPointMulti(new libf::DataPoint(feature_length), lab);
              x_v.push_back(x);
              y_v.push_back(y);
              *mask_ptr = 1;
            }
            mask_ptr+=stride;
            depth_ptr+=stride;
            for(unsigned int l = 0; l < label.size(); l++){
              label_ptr[l]+=stride;
            }
          }
        }
      }
    }

    int feature_position = 0;
    /**************** color patch **********/
    if(_feature_usage[0]){
      int data_index = original_datastorage_size;
      cv::Mat color_b;
      //   cv::GaussianBlur( color, color_b, cv::Size( 3, 3 ), 0, 0 );
      cv::cvtColor(color, color_b, CV_BGR2Lab, CV_8UC3);
      cv::copyMakeBorder(color_b, color_b, _border, _border, _border, _border, cv::BORDER_REFLECT);

      cv::Mat color_patch;

      for(int y = _border; y < color_b.rows - _border; y += stride){
        unsigned short* depth_ptr = depth_im.ptr<unsigned short>(y-_border);
        unsigned char* mask_ptr = mask.ptr<unsigned char>(y-_border);
        for(int x = _border; x < color_b.cols - _border; x += stride){
          if(*mask_ptr){
            float depth =  static_cast<float>(*depth_ptr)/1000.0f;
            int current_size_half = _patch_size/(2.0*depth);
            int current_size = current_size_half*2 + 1;
            cv::resize(color_b(cv::Rect(x-current_size_half,y-current_size_half,current_size,current_size)), color_patch, cv::Size(_patch_size_reduce, _patch_size_reduce));


            //std::vector<float> sum(3,0);
            //for(int yy = 0; yy < _patch_size_reduce; yy++){
            //  unsigned char* patch_ptr = color_patch.ptr<unsigned char>(yy);
            //  for(int xx = 0; xx < _patch_size_reduce; xx++){
            //    for(int cc = 0; cc < 3; cc++){
            //      sum[cc] += patch_ptr[cc];
            //    }
            //    patch_ptr++;
            //  }
            //}
            //sum[0] /= (_patch_size_reduce*_patch_size_reduce);
            //sum[1] /= (_patch_size_reduce*_patch_size_reduce);
            //sum[2] /= (_patch_size_reduce*_patch_size_reduce);


            libf::DataPoint* d = storage.getDataPoint(data_index);
            unsigned char* patch_ptr = color_patch.ptr<unsigned char>(0);
            //int three_count = 0;
            for(int yy = 0; yy < _patch_size_reduce*_patch_size_reduce*3; yy++){
              d->at(yy+feature_position) = *patch_ptr;// - sum[three_count];
              //three_count = (three_count + 1) % 3;
              patch_ptr++;
            }
            data_index++;
          }
          depth_ptr+=stride;
          mask_ptr+=stride;
        }
      }
      feature_position+=_patch_size_reduce*_patch_size_reduce*3;
    }



    /*********** depth *********/
    if(_feature_usage[1]){
      int data_index = original_datastorage_size;
      for(int y = 0; y < depth_im.rows; y+=stride){
        unsigned char* mask_ptr = mask.ptr<unsigned char>(y);
        unsigned short* depth_ptr = depth_im.ptr<unsigned short>(y);
        for(int x = 0; x < depth_im.cols; x+=stride){
          if(*mask_ptr){
            float depth =  static_cast<float>(*depth_ptr)/1000.0f;
            libf::DataPoint* d = storage.getDataPoint(data_index);
            d->at(feature_position) = depth;
            data_index++;
          }
          depth_ptr+=stride;
          mask_ptr+=stride;
        }
      }
      feature_position++;
    }

    //Check if we need the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cld = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    if( _feature_usage[2] || _feature_usage[3]){ //We need the point cloud create it here.
      cld->height = depth_im.rows;
      cld->width = depth_im.cols;
      cld->points.resize(depth_im.rows*depth_im.cols);
      Eigen::MatrixXf mat(3, depth_im.rows*depth_im.cols);
      int index = 0;
      for(int y = 0; y < depth_im.rows; y++){
        for(int x = 0; x < depth_im.cols; x++){
          float depth = static_cast<float>(depth_im.at<unsigned short>(y,x))/1000.0f;
          if( depth < d_min || depth > d_max){
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
      }
      index = 0;
      Eigen::MatrixXf rect = (c._extrinsic.linear()* c._intrinsic_inverse)*mat + c._extrinsic.translation().rowwise().replicate(depth_im.rows*depth_im.cols);
      for(int y = 0; y < depth_im.rows; y++){
        for(int x = 0; x < depth_im.cols; x++){
          cld->points[index].x = rect(0, index);
          cld->points[index].y = rect(1, index);
          cld->points[index].z = rect(2, index);
          index++;
        }
      }
    }


    /*********** height *********/
    if(_feature_usage[2]){
      int data_index = original_datastorage_size;
      for(int y = 0; y < depth_im.rows; y+=stride){
        unsigned char* mask_ptr = mask.ptr<unsigned char>(y);
        for(int x = 0; x < depth_im.cols; x+=stride){
          if(*mask_ptr){
            float height =  cld->points[y*depth_im.cols +x].z;
            libf::DataPoint* d = storage.getDataPoint(data_index);
            d->at(feature_position) = height;
            data_index++;
          }
          mask_ptr+=stride;
        }
      }
      feature_position++;
    }

    //check if we need normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    if(_feature_usage[3]){
      pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
      ne.setMaxDepthChangeFactor(0.02f);
      ne.setNormalSmoothingSize(10.0f);
      ne.setInputCloud(cld);
      ne.compute(*normals);
    }

    /********* normal **********/
    if(_feature_usage[3]){

      int data_index = original_datastorage_size;
      int index = 0;
      for(int y = 0; y < depth_im.rows; y+=stride){
        unsigned char* mask_ptr = mask.ptr<unsigned char>(y);
        for(int x = 0; x < depth_im.cols; x+=stride){
          if(*mask_ptr){
            index =  y*depth_im.cols +x;
            libf::DataPoint* d = storage.getDataPoint(data_index);
            if(isnan(normals->points[index].normal_x)){
              d->at(feature_position) = -2;
              // d->at(feature_position+1) = -2;
              // d->at(feature_position+2) = -2;
            }else{
              // d->at(feature_position) = normals->points[index].normal_x;
              // d->at(feature_position+1) = normals->points[index].normal_y;
              // d->at(feature_position+2) = normals->points[index].normal_z;
              d->at(feature_position) = acos(fabs(normals->points[index].normal_z));
            }
            data_index++;
          }
          mask_ptr+=stride;
        }
      }
      feature_position+=1;
    }


    /*
     *  cv::Mat normal_image(color.rows, color.cols, CV_32FC3);
     *  index = 0;
     *  for(int y = 0; y < color.rows; y++){
     *    float* n_ptr = normal_image.ptr<float>(y);
     *    for(int x = 0; x < color.cols; x++){
     *      n_ptr[0] = normals->points[index].normal_x;
     *      n_ptr[1] = normals->points[index].normal_y;
     *      n_ptr[2] = normals->points[index].normal_z;
     *      n_ptr+=3;
     *      index++;
               }
               }

               cv::copyMakeBorder(normal_image, normal_image, border, border, border, border, cv::BORDER_REFLECT);


               for(int y = border; y < color_b.rows - border; y += stride){
                 for(int x = border; x < color_b.cols - border; x += stride){
                   cv::Mat temp;

                   //           h.at<float>(y-border,x-border) = rect(2);
                   index = (y-border)*depth_im.cols + x -border;
                   if(isnan(cld->points[index].x)) continue;
                   int l = label.at<label_type>(y-border, x-border);
                   if(train && (l < 0)) continue;
                   float depth =  static_cast<float>(depth_im.at<unsigned short>(y-border,x-border))/1000.0f;
                   int current_size_half = patch_size/(2.0*depth);
                   int current_size = current_size_half*2 + 1;
                   cv::resize(color_b(cv::Rect(x-current_size_half,y-current_size_half,current_size,current_size)), temp, cv::Size(patch_size_reduce, patch_size_reduce));

                   cv::Mat temp_norm;
                   cv::resize(normal_image(cv::Rect(x-current_size_half,y-current_size_half,current_size,current_size)), temp_norm, cv::Size(patch_size_reduce, patch_size_reduce));


                   int further_features = 5;
                   libf::DataPoint* data  = new libf::DataPoint(temp.cols*temp.rows*4+further_features);
                   data->at(0) = cld->points[index].z;
                   data->at(1) = depth;
                   if(isnan(normals->points[index].normal_x)){
                     data->at(2) = -2;
                     data->at(3) = -2;
                     data->at(4) = -2;
               }else{
                 data->at(2) = normals->points[index].normal_x;
                 data->at(3) = normals->points[index].normal_y;
                 data->at(4) = normals->points[index].normal_z;
               }
               for(int yy = 0; yy < temp.cols*temp.rows*3; yy++){
                 data->at(yy+further_features) = temp.ptr<uchar>(0)[yy];
               }
               float a_n = normals->points[index].normal_x;
               float b_n = normals->points[index].normal_y;
               float c_n = normals->points[index].normal_z;
               index = temp.cols*temp.rows*3+further_features;
               //       int x_nn, y_nn;
               //       int patch_size_reduce_half = patch_size_reduce/2;
               //       for(int y_n = y-patch_size_reduce_half ; y_n <= y+patch_size_reduce_half; y_n++){
               //         for(int x_n = x-patch_size_reduce_half ; x_n <= x+patch_size_reduce_half; x_n++){
               //           x_nn = x_n;
               //           y_nn = y_n;
               //           if(x_nn < 0) x_nn = 0;
               //           if(x_nn >= color.cols) x_nn = color.cols -1;
               //           if(y_nn < 0) y_nn = 0;
               //           if(y_nn >= color.rows) y_nn = color.rows -1;
               //           data->at(index) = a_n * normals->points[y_nn * color.cols + x_nn].normal_x +
               //                             b_n * normals->points[y_nn * color.cols + x_nn].normal_y +
               //                             c_n * normals->points[y_nn * color.cols + x_nn].normal_z;
               //           if(isnan(data->at(index))) data->at(index) = -2;
               //           index++;
               //
               //         }
               //       }
               float* norm_patch_ptr = temp_norm.ptr<float>(0);
               for(int yy = 0; yy < temp.cols*temp.rows; yy++){
                 data->at(index) = a_n * norm_patch_ptr[0] + b_n * norm_patch_ptr[1] + c_n * norm_patch_ptr[2];
                 if(isnan(data->at(index))) data->at(index) = -2;
                 norm_patch_ptr++;
                 index++;
               }

               storage.addDataPoint(data, l);
               x_v.push_back(x-border);
               y_v.push_back(y-border);
               }
               }
               */

    //    cv::Laplacian(color_b, color_laplace, CV_8UC3);
               }

private:
  int _patch_size;
  int _patch_size_reduce;
  int _border;
  std::vector<bool> _feature_usage;

};
}