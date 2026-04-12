#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include <memory>
#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "Ilayer.h"
#include "Matrix.h"
#include "Activation.h"
#include "Acts.h"

namespace cobalt_715::nn{

//全結合層
//層に必要な重み、バイアス、活性化関数などを保持する
//ミニバッチ対応
struct DenseLayer : public ILayer{
  const Matrix *input_ptr;//逆伝播で必要なため順伝播時この層への入力を見る
  Matrix W,b;//重み、バイアス
  Matrix Wt;//転置済み重みを保持しメモリ確保を減らす
  Matrix z,a;//活性化なし、活性化済み
  Matrix dW,db;//重みの微分、バイアスの微分
  Matrix delta,grad;//この層での誤差、次の層に渡す勾配
  Matrix delta_t;//転置済みdeltaを保持しメモリ確保をへらす

  const Activation *act = &activations::LeakyReLU;//活性化関数とその微分。デフォルトではLeakyReLU

  //順伝播
  //前層の出力を受け取る
  const Matrix& forward(const Matrix& input) override{
    input_ptr = &input;

    //バッチサイズにより変わる部分なので分岐している
    if(input.cols() != z.cols()){
      z = Matrix(W.rows(),input.cols());
      a = Matrix(W.rows(),input.cols());
    }

    Matrix::dot_Bt(W,input.T(),z);//z = W  * input

    //バイアスを足している
    //NNのバイアスは行で見て等しいという性質があるため専用関数にしている
    add_bias();

    activation();//活性化している

    return a;
  }

  //逆伝播
  //次の層の勾配を受け取る
  const Matrix& backward(const Matrix& grad_output) override{
    delta_hadamard_add_db(grad_output);//要素積とdb更新を同時にしている
    Matrix::dot_Bt(delta,*input_ptr,dW);//dW = delta * a転置
    W.T_to(Wt);//メモリ確保を抑えるためWの転置をWtに保存している

    //バッチサイズにより変わる部分なので分岐している
    if(grad.cols() != delta.cols() || grad.rows() != Wt.rows())
      grad = Matrix(Wt.rows(),delta.cols());

    //バッチサイズにより変わる部分なので分岐している
    if(delta.rows() != delta_t.cols() || delta.cols() != delta_t.rows())
      delta_t = Matrix(delta.cols(),delta.rows());

    delta.T_to(delta_t);//メモリ確保を抑えるため

    Matrix::dot_Bt(Wt,delta_t,grad);//grad = W転置 * delta

    return grad;
  }

  const Matrix& backward_no_grad(const Matrix& grad_output) override{
    delta_hadamard_add_db(grad_output);//要素積とdb更新を同時にしている
    Matrix::dot_Bt(delta,*input_ptr,dW);//dW = delta * a転置

    return grad;
  }

  //NNのバイアスは行で見て等しいという性質があるため専用関数にしている
  void add_bias(){
    double *zd = z.data().data();
    double *bd = b.data().data();

    const int zrows = z.rows();
    const int zcols = z.cols();

    for(int i = 0;i < zrows;i++){
      double bias = bd[i];
      for(int j = 0;j < zcols;j++){
        zd[i * zcols + j] += bias;
      }
    }
  }

  //zの要素を活性化させてaに保持する
  void activation(){
    const int zrows = z.rows();
    const int zcols = z.cols();

    const double *zd = z.data().data();
    double *ad = a.data().data();

    int size = z.rows() * z.cols();
    for(int i = 0;i < size;i++){
      ad[i] = act->act(zd[i]);
    }
  }

  //要素積とdb更新を同時にしている
  void delta_hadamard_add_db(const Matrix &loss){
    const int lrows = loss.rows();
    const int lcols = loss.cols();

    if(delta.cols() != lcols || delta.rows() != lrows) delta = Matrix(lrows,lcols);//サイズが違うときだけ再確保

    double *dd = delta.data().data();
    const double *ld = loss.data().data();
    const double *zd = z.data().data();
    const double *ad = a.data().data();

    double *dbd = db.data().data();

    for(int i = 0;i < lrows;i++){
      const int ilcols = i * lcols;
      double total_db_element = 0;

      for(int j = 0;j < lcols;j++){
        dd[ilcols + j] = ld[ilcols + j] * act->d_act(zd[ilcols + j],ad[ilcols + j]);
        total_db_element += dd[ilcols + j];
      }
      dbd[i] = total_db_element;
    }
  }

  void step(double lr,int batch_size) override{
    Matrix::scale(dW,lr,dW);//dW = dW * lr;
    Matrix::subtract(W,dW,W);//W -= dW;

    Matrix::scale(db,lr,db);//l.db = l.db * lr;
    Matrix::subtract(b,db,b);//b -= db;
  }

  virtual std::string get_type() const override{
    return "dense";
  }

  std::string to_string() const override{
    std::string s;
    s += "activation " + act->name;
    s += "\nW\n";
    s += W.to_string() + "\nb\n";
    s += b.to_string();
    return s;
  }

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json() const override{
    nlohmann::ordered_json j;
    j["layer_type"] = get_type();
    j["in"] = W.cols();
    j["out"] = W.rows();
    j["activation"] = act->name;
    j["weight"] = W.data();
    j["bias"] = b.data();
    return j;
  }

  //json形式で層をセットするとき使う
  void load_json(nlohmann::ordered_json j) override{
  }

  void random_init(std::mt19937 &gen) override{
    double limit = sqrt(2.0 / (W.rows() + W.cols()));
    std::uniform_real_distribution<double> dist(-limit,limit);
    for(double &d:W.data()){
      d = dist(gen);
    }

    for(double &d:b.data()){
      d = limit / (W.rows() + W.cols());
    }
  }

  DenseLayer(int in,int out)
    : W(out,in),
      b(out,1),
      Wt(in,out),
      z(out,1),
      a(out,1),
      dW(out,in),
      db(out,1),
      delta(1,1),
      grad(1,1),
      delta_t(1,1){}

  DenseLayer(nlohmann::ordered_json j) : DenseLayer(j["in"].get<int>(),j["out"].get<int>()){
    act = &Acts::getAct(j["activation"]);

    W = Matrix(W.rows(),W.cols(),j["weight"].get<std::vector<double>>());
    b = Matrix(b.rows(),b.cols(),j["bias"].get<std::vector<double>>());
  }
};

}//namespace cobalt_715::nn

#endif