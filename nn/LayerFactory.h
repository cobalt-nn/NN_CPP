#ifndef LAYERFACTORY_H
#define LAYERFACTORY_H

#include <unordered_map>
#include <functional>
#include <memory>
#include <string>
#include "nlohmann/json.hpp"
#include "ILayer.h"
#include "DenseLayer.h"

namespace cobalt_715::nn{

//jsonを渡して対応する層のインスタンスを生成できるようにする
class LayerFactory{
public:
  using Creator = std::function<std::unique_ptr<ILayer>(const nlohmann::json&)>;

private:
  inline static std::unordered_map<std::string,Creator> factory;

  //標準の層を登録する
  static void init(){
    static bool initialized = [](){

      register_layer("dense",
        [](const nlohmann::json& j){
          return std::make_unique<DenseLayer>(j);
        }
      );

      return true;
    }();
  }

public:
  static void register_layer(const std::string& name, Creator c);
  static std::unique_ptr<ILayer> create(const nlohmann::json& j);
};

}//namespace cobalt_715::nn

#endif