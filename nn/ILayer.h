#ifndef ILAYER_H
#define ILAYER_H

#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "Matrix.h"

namespace cobalt_715::nn{

//層の基底クラス
//全結合やCNNを同じTrainerで扱えるようにしている
struct ILayer{
  //順伝播
  //前層の出力を受け取る
  virtual const Matrix& forward(const Matrix& input) = 0;

  //逆伝播
  //次の層の勾配を受け取る
  virtual const Matrix& backward(const Matrix& grad_output) = 0;

  //逆伝播の時最後の層は勾配計算がいらないため
  virtual const Matrix& backward_no_grad(const Matrix& grad_output){
    return backward(grad_output);
  }

  //更新
  //学習率、バッチサイズを受け取る
  virtual void step(double lr,int batch_size=64) = 0;

  //層の種類を返す。適切にオーバーライドすること
  virtual std::string get_type() const = 0;

  //文字列にしたいとき使う
  virtual std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  //json形式で保存するとき使う
  virtual nlohmann::ordered_json to_json() const = 0;

  //json形式で層をセットするとき使う
  virtual void load_json(nlohmann::ordered_json j) = 0;

  //ランダム初期化する
  virtual void random_init(std::mt19937 &gen){}

  virtual ~ILayer(){}
};

}//namespace cobalt_715::nn

#endif