#pragma once

#include <exception>
#include <fstream>
#include <map>
#include <string>


#include "json/json.h"

namespace Utils {

class KeyNotFoundException : public std::exception{
public:

  explicit KeyNotFoundException(std::string key): _missing_key(key) { }

  virtual char const* what() const noexcept (true) {
    std::string msg = "No entry for " + _missing_key + " found.";
    return msg.c_str();
  }
private:
  std::string _missing_key;
};

class Config{
public:
  Config(std::string config_file, std::map<std::string, std::string> new_params, std::string root_dir_key = "root_dir");

  //Mainly only here for testing.
  Config(std::map<std::string, std::string> new_params);

  std::string getPath(std::string key) const;

  std::string getJsonValueAsString(std::string key) const;

  Json::Value getRaw(std::string key) const;

  template<typename T>
  T getFromFile(std::string key) const{
    std::string file_name = getPath(key);
    if(file_name.find(".json") != std::string::npos){
      Config temp_conf(file_name);
      return temp_conf.get<T>(key);
    }else{
      throw std::runtime_error("There was no valid json file for the key: " + key + "\ngot: " + file_name);
    }
  }

  template<typename T>
  T get(std::string key, const T& default_val) const{
    //T::unimplemented_function; //Will throw an compilation error if get is called with a unspecialized template type.
    try{
      return get<T>(key);
    }catch(KeyNotFoundException e){
      return default_val;
    }
  }

  template<typename T>
  T get(std::string key) const{
    T::unimplemented_function; //Will throw an compilation error if get is called with a unspecialized template type.
  }

private:
  Config(std::string file_name);

private:
  Json::Value _conf;
  std::string _root_dir;
};

template<>
bool Config::get<bool>(std::string key) const;

template<>
std::vector<bool> Config::get<std::vector<bool> >(std::string key) const;

template<>
double Config::get<double>(std::string key) const;

template<>
std::vector<double> Config::get<std::vector<double> >(std::string key) const;

template<>
float Config::get<float>(std::string key) const;

template<>
std::vector<float> Config::get<std::vector<float> >(std::string key) const;

template<>
int Config::get<int>(std::string key) const;

template<>
std::vector<int> Config::get<std::vector<int> >(std::string key) const;

template<>
std::string Config::get<std::string>(std::string key) const;

template<>
std::vector<std::string> Config::get<std::vector<std::string> >(std::string key) const;

template<>
unsigned int Config::get<unsigned int>(std::string key) const;

template<>
std::vector<unsigned int> Config::get<std::vector<unsigned int> >(std::string key) const;
}