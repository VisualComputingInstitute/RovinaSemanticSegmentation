#pragma once

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

// STL includes
#include <string>

class Calibration {
public:

  Calibration();

  Calibration(std::string filename);

  void saveToFile(std::string json_filename);

  std::string _filename;
  Eigen::Matrix3f _intrinsic;
  Eigen::Matrix3f _intrinsic_inverse;
  Eigen::Isometry3f _extrinsic;
};