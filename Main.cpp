#include <iostream>
#include <vector>
#include "nn/Matrix.h"
#include "nn/Trainer.h"
#include "nn/Activation.h"
#include "nn/Acts.h"

int main(){
  cobalt_715::nn::Matrix xor_input1(2,1);//2 * 1行列を要素をすべて0で初期化
  std::cout << xor_input1.to_string() << std::endl;//行列の内容を出力

  std::vector<double> v1 = {0,1};
   cobalt_715::nn::Matrix xor_input2(2,1,v1);//2 * 1行列の中身をv1にして初期化
  std::cout << xor_input2.to_string() << std::endl;

  std::vector<double> v2 = {
                           {1},
                           {0}
                           };

  cobalt_715::nn::Matrix xor_input3(2,1,v2);//２次元配列をそのまま行列になる
  std::cout << xor_input3.to_string() << std::endl;

  cobalt_715::nn::Matrix xor_input4(2,1,1);//第三引数の数値で行列を初期化
  std::cout << xor_input4.to_string() << std::endl;

  std::vector<cobalt_715::nn::Matrix> xor_inputs = {xor_input1,xor_input2,xor_input3,xor_input4};//ミニバッチ化するために列ベクトルをまとめる
  std::vector<cobalt_715::nn::Matrix> xor_batch_inputs = cobalt_715::nn::Trainer::make_batches(xor_inputs);//バッチ化されたMatrix配列を受け取る
  std::cout << xor_batch_inputs.at(0).to_string() << std::endl;

  std::vector<double> xor_targets = {0,1,1,0};
  std::vector<cobalt_715::nn::Matrix> xor_batch_targets = {cobalt_715::nn::Matrix(1,4,xor_targets)};//xorの教師をバッチに合わせて初期化。xor_batch_inputsの列とこれの列が対応している

  cobalt_715::nn::Trainer t = cobalt_715::nn::Trainer("nn/models/model.json");//モデルの場所を指定している

  std::vector<int> model_size = {2,4,1};//モデルのサイズを指定する
  std::vector<cobalt_715::nn::Activation> acts = {cobalt_715::nn::Acts::getAct("LeakyReLU"),cobalt_715::nn::Acts::getAct("LeakyReLU")};//各層の活性化関数を指定する

  t.load_model();//Trainer生成時に指定したtxtファイルからモデル読み込み
  t.make_dense_model(model_size,acts);//モデルを生成

  for(int i = 0;i < 10000;i++){//10000回学習
    if(i % 1000 == 0){
      std::cout << i << "epoch\n";
      std::cout << t.forward_network(xor_batch_inputs.at(0)).to_string() << std::endl;//順伝播
    }
    t.learn_batch(xor_batch_inputs,xor_batch_targets);//ミニバッチ済みの配列を受け取る。内部でミニバッチ化するものもある
  }

  t.save_model();//Trainer生成時に指定したtxtファイルにモデル読書き込み

  return 0;
}
