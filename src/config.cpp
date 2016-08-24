// STL inlcudes
#include <iostream>

// Local includes
#include "config.h"

using namespace Utils;

Config::Config(std::string config_file, std::map<std::string, std::string> new_params, std::string root_dir_key){

  //Try to open the config file.
  std::ifstream conf_file(config_file);
  if(!conf_file){
    throw std::runtime_error("Failed to open config file " + config_file);
  }

  //Try to parse it.
  Json::Reader reader;
  if(! reader.parse(conf_file, _conf)){
    throw std::runtime_error("Failed to parse config file " + config_file + "\n" + reader.getFormatedErrorMessages());
  }

  //Check if the other parameters specified by the map are set or not.
  for(std::pair<std::string, std::string> c : new_params){
    Json::Value temp;
    reader.parse(c.second, temp);
    _conf[c.first] = temp;
  }
  _root_dir = get<std::string>(root_dir_key);
}

Config::Config(std::map<std::string, std::string> new_params){
  //Only load stuff from the map.
  Json::Reader reader;
  for(std::pair<std::string, std::string> c : new_params){
    Json::Value temp;
    reader.parse(c.second, temp);
    _conf[c.first] = temp;
  }
  _root_dir = "";
}

Config::Config(std::string file_name){
  //Try to open the config file.
  std::ifstream conf_file(file_name);
  if(!conf_file){
    throw std::runtime_error("Failed to open file " + file_name);
  }

  //Try to parse it.
  Json::Reader reader;
  if(! reader.parse(conf_file, _conf)){
    throw std::runtime_error("Failed to parse file " + file_name + "\n" + reader.getFormatedErrorMessages());
  }
}


std::string Config::getPath(std::string key) const{
  return _root_dir + "/" + get<std::string>(key);
}

std::string Config::getJsonValueAsString(std::string key) const{
  return _conf[key].toStyledString();
}

Json::Value Config::getRaw(std::string key) const{
  return _conf[key];
}



template<>
bool Config::get<bool>(std::string key) const {
  if(_conf.isMember(key)){
    return _conf[key].asBool();
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::vector<bool> Config::get<std::vector<bool> >(std::string key) const {
  if(_conf.isMember(key)){
    std::vector<bool> result;
    for(const Json::Value& p : _conf[key]) {
      result.push_back(p.asBool());
    }
    return result;
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
double Config::get<double>(std::string key) const {
  if(_conf.isMember(key)){
    return _conf[key].asDouble();
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::vector<double> Config::get<std::vector<double> >(std::string key) const {
  if(_conf.isMember(key)){
    std::vector<double> result;
    for(const Json::Value& p : _conf[key]) {
      result.push_back(p.asDouble());
    }
    return result;
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
float Config::get<float>(std::string key) const {
  if(_conf.isMember(key)){
    return _conf[key].asFloat();
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::vector<float> Config::get<std::vector<float> >(std::string key) const {
  if(_conf.isMember(key)){
    std::vector<float> result;
    for(const Json::Value& p : _conf[key]) {
      result.push_back(p.asFloat());
    }
    return result;
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
int Config::get<int>(std::string key) const {
  if(_conf.isMember(key)){
    return _conf[key].asInt();
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::vector<int> Config::get<std::vector<int> >(std::string key) const {
  if(_conf.isMember(key)){
    std::vector<int> result;
    for(const Json::Value& p : _conf[key]) {
      result.push_back(p.asInt());
    }
    return result;
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::string Config::get<std::string>(std::string key) const {
  if(_conf.isMember(key)){
    return _conf[key].asString();
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::vector<std::string> Config::get<std::vector<std::string> >(std::string key) const {
  if(_conf.isMember(key)){
    std::vector<std::string> result;
    for(const Json::Value& p : _conf[key]) {
      result.push_back(p.asString());
    }
    return result;
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
unsigned int Config::get<unsigned int>(std::string key) const {
  if(_conf.isMember(key)){
    return _conf[key].asDouble();
  }else{
    throw KeyNotFoundException(key);
  }
}

template<>
std::vector<unsigned int> Config::get<std::vector<unsigned int> >(std::string key) const {
  if(_conf.isMember(key)){
    std::vector<unsigned int> result;
    for(const Json::Value& p : _conf[key]) {
      result.push_back(p.asUInt());
    }
    return result;
  }else{
    throw KeyNotFoundException(key);
  }
}


