#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include <memory>
#include "Matrix.h"
#include "ILayer.h"
#include "DenseLayer.h"
#include "Activation.h"

namespace cobalt_715::nn{

//ニューラルネットワークのモデルを保持し学習、推論を行うクラス
//Layer.hのインスタンスによって構成される配列をモデルとして持つ
//自乗誤差の総和/2を損失関数として主に採用しているlearnは損失関数を変えられないがbackwardは勾配を受け取るため、ある程度自由に損失関数を設定できる
//全結合はミニバッチ対応
struct Trainer{
private:
  std::string modelPath;//json形式になっている。モデルを読み書きするテキストファイルを指定する。例"nn/models/model.json"など。コンストラクタで指定する
  std::vector<std::unique_ptr<ILayer>> layers;//モデル
  const double lr = 0.1;//学習率

public:
  //順伝播。入力を受け取る
  Matrix forward_network(const Matrix &x);

  //逆伝播。勾配、入力、勾配を返すかどうかの条件の順で受け取る
  //勾配を直接受け取ることに注意が必要
  //forwardで得られる情報が必要なため基本的にはbackward(forward(input) - target),input,false)のように使う
  //バッチサイズに合わせるための分岐とメモリ確保をしているため少し遅い
  //compute_input_grad = falseなら1 * 1の行列を返す
  //compute_input_grad = trueなら最初の層からの勾配を返す
  Matrix backward_network(const Matrix &dout,bool compute_input_grad = false);

  //学習の工程をまとめた
  //flattenつまり列ベクトルにした入力をまとめたものと、同じくflattenした教師をまとめたものとバッチサイズを受け取る
  //自乗誤差の総和/2として損失を計算する
  void learn(const std::vector<Matrix> &inputs,const std::vector<Matrix> &targets,const int batch_size=64);

  //バッチ済みのものを受け取る
  void learn_batch(const std::vector<Matrix> &inputs,const std::vector<Matrix> &targets);

  //列ベクトルをまとめたものからバッチサイズに合わせたものを返す
  static std::vector<Matrix> make_batches(const std::vector<Matrix> &matrixs,const int batch_size=64);

  //パラメーターを更新する
  //学習させたいときは必ず呼び出す。learn,learn_batchの中ではすでに呼び出している
  void step_network(int batch_size=64){
    for(auto &l:layers){
      l->step(lr,batch_size);
    }
  }

  //以下はモデルのセーブ、ロードをまとめたもの
  //this->modelPathで示されるテキストファイルで読み書きする

  //テキストファイルからモデルを読み込む。makeModelからは呼ばれないため使う前に必ず呼び出すこと
  void load_model();

  //テキストファイルにモデルを書き込む。learnなどからは呼ばれないため保存したいときは必ず呼び出すこと
  void save_model() const;

  //全結合用のNNを作る
  //makeModelはモデルのサイズや活性化関数を指定できランダム初期値にする
  //例えばstd::vector<int> a = {784,256,64,10};を引数に渡すと層が3つでき、最初の層は784次元の入力から256次元の出力、最後の層は64次元の入力から10次元の出力になるようにモデルが生成される
  //より正確には列がindex + 1,行がindexとなる重みと列がindex + 1,行が1のバイアスが作られる
  //活性化関数はデフォルトでLeakyReLU
  void make_dense_model(const std::vector<int> &a,int seed = 0){
    std::vector<Activation> acts;
    for(int i = 0;i < a.size() - 1;i++){
      acts.push_back(activations::LeakyReLU);
    }
    make_dense_model(a,acts,seed);
  }

  //活性化関数も指定できるmake_dense_model。層の生成アルゴリズムによりa.size() - 1 == acts.size()にあたる活性化関数配列を受け取る
  //活性化関数はActivation.hファイルにより詳しいことを書いている
  void make_dense_model(const std::vector<int> &a,const std::vector<Activation> &acts,int seed = 0);

  //モデルを読み書きするためのテキストファイル、学習率を受け取る
  Trainer(const std::string &txt,const double d=0.01) : modelPath(txt),lr(d){}

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json() const{
    nlohmann::ordered_json j;
    nlohmann::ordered_json j_array = nlohmann::ordered_json::array();

    for(const auto &l:layers){
      j_array.push_back(l->to_json());
    }

    j["layers"] = j_array;

    return j;
  }
};

}//namespace cobalt_715::nn

#endif