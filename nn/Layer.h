#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include "Matrix.h"
#include "Activation.h"

namespace cobalt_715::nn{

//NNの層に必要な重み、バイアス、活性化関数などを保持する
struct Layer{
  Matrix W,b;//重み、バイアス
  Matrix Wt;//転置済み重みを保持しメモリ確保を減らす
  Matrix z,a;//活性化なし、活性化済み
  Matrix dW,db;//重みの微分、バイアスの微分
  Matrix delta,grad;//この層での誤差、次の層に渡す勾配
  Matrix delta_t;//転置済みdeltaを保持しメモリ確保をへらす

  const Activation *act = &activations::LeakyReLU;//活性化関数とその微分。デフォルトではLeakyReLU

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

    if(a.rows() != zrows || a.cols() != zcols) a = Matrix(zrows,zcols);//サイズが違うときだけ再確保

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

  Layer(int in,int out)
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

  std::string to_string() const{
    std::string s;
    s += "activation " + act->name;
    s += "\nW\n";
    s += W.to_string() + "\nb\n";
    s += b.to_string();
    return s;
  }
};

}//namespace cobalt_715::nn

#endif