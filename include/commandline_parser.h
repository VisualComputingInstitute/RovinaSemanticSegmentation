#pragma once

#include <map>
#include <stdexcept>
#include <string>

namespace Utils{

  bool parseParamters(int argc, char** argv,
                      std::map<std::string, std::string>& param_name_val){

  //Assume that we get parameters of the form "--param1 val1 ..."
  bool parse_param = true;
  std::string param_name;
  for(int i = 1; i < argc; i++){
    std::string parameter = std::string(argv[i]);
    if(parse_param){
      //check if we get a string of the form "--****"
      if(parameter.substr(0, 2) == "--"){
        param_name = parameter.substr(2, std::string::npos);
      }else{
        return false;
      }
    }else{
      param_name_val[param_name] = parameter;
    }
    parse_param = !parse_param;
  }
  if(!parse_param){
    throw std::runtime_error("Missing value for the option: " + param_name);
  }
  return true;
}
}