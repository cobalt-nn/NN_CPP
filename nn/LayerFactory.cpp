#include <memory>
#include <string>
#include "LayerFactory.h"

namespace cobalt_715::nn{

void LayerFactory::register_layer(const std::string& name, Creator c){
  factory.emplace(name,c);
}

std::unique_ptr<ILayer> LayerFactory::create(const nlohmann::json& j){
  init();
  auto it = factory.find(j["layer_type"]);

  if(it != factory.end()){
    return it->second(j);
  }

  throw std::runtime_error("Unknown layer type: " + j["layer_type"].get<std::string>());
}

}//namespace cobalt_715::nn