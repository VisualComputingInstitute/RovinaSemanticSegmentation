// Local includes
#include "calibration.h"

// STL includes
#include <fstream>
#include <stdexcept>
#include <vector>

// Third party includes
#include "json/json.h"


Calibration::Calibration(){
}

Calibration::Calibration(std::string filename) : _filename(filename) {
  Json::Value calib;

  std::ifstream calib_file(_filename);
  if(!calib_file){
    throw std::runtime_error("Failed to open calibration file " + _filename);
  }

  //Try to parse it.
  Json::Reader reader;
  if(! reader.parse(calib_file, calib)){
    throw std::runtime_error("Failed to parse calibration file " + _filename + "\n" + reader.getFormatedErrorMessages());
  }
  if(!calib.isMember("intrinsic") || !calib.isMember("translation") || !calib.isMember("rotation")){
    throw std::runtime_error("Calibration file " + _filename + " is not complete!");
  }

  //intrinsics
  for(unsigned int index = 0; index < 9; index++) {
    _intrinsic(index) = calib["intrinsic"][index].asFloat();
  }
  _intrinsic.transposeInPlace();
  _intrinsic_inverse = _intrinsic.inverse();

  //Rotation
  if(calib["rotation"]["format"].asString() == "q3"){
    float qx, qy, qz, qw;
    qx = calib["rotation"]["data"][0].asFloat();
    qy = calib["rotation"]["data"][1].asFloat();
    qz = calib["rotation"]["data"][2].asFloat();
    qw = sqrt(1.0 - qx*qx - qy*qy - qz*qz);
    Eigen::Quaternion<float> q;
    q.w() = qw;
    q.z() = qz;
    q.y() = qy;
    q.x() = qx;
    _extrinsic.linear() = q.matrix();
  }else if(calib["rotation"]["format"].asString() == "q4"){
    Eigen::Quaternion<float> q;
    q.x() = calib["rotation"]["data"][0].asFloat();
    q.y() = calib["rotation"]["data"][1].asFloat();
    q.z() = calib["rotation"]["data"][2].asFloat();
    q.w() = calib["rotation"]["data"][3].asFloat();
    _extrinsic.linear() = q.matrix();
  }else if(calib["rotation"]["format"].asString() == "r3"){
    Eigen::Matrix3f tmp;
    for(unsigned int i = 0; i < 9; i++){
      tmp(i) = calib["rotation"]["data"][i].asFloat();
    }
    _extrinsic.linear() = tmp;
  }

  //Translation
  Eigen::Vector3f t;
  t(0) = calib["translation"][0].asFloat();
  t(1) = calib["translation"][1].asFloat();
  t(2) = calib["translation"][2].asFloat();
  _extrinsic.translation() = t;
}

void Calibration::saveToFile(std::string json_filename){
  std::ofstream file_id;
  file_id.open(json_filename);
  Json::Value calib;

  Json::Value intrinsic(Json::arrayValue);
  Eigen::Matrix3f tmp = _intrinsic.transpose();
  for(unsigned int index = 0; index < 9; index++){
    intrinsic.append(Json::Value(tmp(index)));
  }
  calib["intrinsic"] = intrinsic;

  Json::Value translation(Json::arrayValue);
  translation.append(Json::Value(_extrinsic.translation()(0)));
  translation.append(Json::Value(_extrinsic.translation()(1)));
  translation.append(Json::Value(_extrinsic.translation()(2)));
  calib["translation"] = translation;


  calib["rotation"]["format"] = "r3";
  Json::Value rotation(Json::arrayValue);
  tmp = _extrinsic.linear();//.transpose();
  for(unsigned int index = 0; index < 9; index++){
    rotation.append(Json::Value(tmp(index)));
  }
  calib["rotation"]["data"] = rotation;


  Json::StyledWriter styledWriter;
  file_id << styledWriter.write(calib);

  file_id.close();
}