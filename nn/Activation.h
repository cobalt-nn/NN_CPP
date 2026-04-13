#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <string>
#include <cmath>

namespace cobalt_715::nn{

//活性化関数とその微分を保持する
struct Activation{
  const std::string name;//活性化関数名
  double (*act)(double);//活性化関数
  double (*d_act)(double z,double a);//微分。様々な微分に対応するため微分に必要な情報を活性化前、活性化後の順で受け取る
};

//基本的な活性化関数をまとめている
namespace activations{

inline const Activation Sigmoid{
  "Sigmoid",
  [](double x){
    return 1.0 / (1.0 + std::exp(-x));
  },
  [](double z,double a){
    return a * (1.0 - a);
  }
};

inline const Activation ReLU{
  "ReLU",
  [](double x){
    return (0.0 < x) ? x:0.0;
  },
  [](double z,double a){
    return (0.0 < z) ? 1.0:0.0;
  }
};

inline const Activation LeakyReLU{
  "LeakyReLU",
  [](double x){
    return (0.0 < x) ? x:0.01 * x;;
  },
  [](double z,double a){
    return (0.0 < z) ? 1.0:0.01;
  }
};

inline const Activation Straight_Through_Estimator{
  "Straight_Through_Estimator",
  [](double x){
    return (0.0 < x) ? 1.0:0.0;
  },
  [](double z,double a){
    return 1.0;
  }
};

}//namespace activations

}//namespace cobalt_715::nn

#endif