#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include "Trainer.h"
#include "Acts.h"
#include "LayerFactory.h"

namespace cobalt_715::nn{

Matrix Trainer::forward_network(const Matrix &x){
  const Matrix *out = &x;

  for(auto &l:layers){
    out = &l->forward(*out);
  }

  return *out;
}

Matrix Trainer::backward_network(const Matrix &dout,bool compute_input_grad){
  const Matrix *out = &dout;

  for(int i = layers.size() - 1;i >= 1;i--){
    out = &layers[i]->backward(*out);
  }

  if(compute_input_grad){
    out = &layers[0]->backward(*out);
  }else{
    out = &layers[0]->backward_no_grad(*out);
  }

  return *out;
}

//flattenつまり1列の行列の配列を受け取る
void Trainer::learn(const std::vector<Matrix> &inputs,const std::vector<Matrix> &targets,const int batch_size){
  learn_batch(make_batches(inputs,batch_size),make_batches(targets,batch_size));
}

//ミニバッチ化したMatrixの配列を受け取る
void Trainer::learn_batch(const std::vector<Matrix> &inputs,const std::vector<Matrix> &targets){
  const int input_rows = inputs.at(0).rows();
  const int target_rows = targets.at(0).rows();
  #ifndef NDEBUG
  if(inputs.size() != targets.size())
  throw std::invalid_argument(
    "learn_batch(): number of input samples and target samples must match. "
    "inputs.size()=" + std::to_string(inputs.size()) +
    ", targets.size()=" + std::to_string(targets.size())
  );

  for(int i = 0;i < inputs.size();i++){
    const Matrix &m = inputs.at(i);
    if(m.rows() != input_rows)
      throw std::invalid_argument(
        "learn_batch(): all input samples must have the same number of rows. "
        "Sample index: " + std::to_string(i) +
        ", expected rows: " + std::to_string(input_rows) +
        ", actual rows: " + std::to_string(m.rows())
      );
  }

  for(int i = 0;i < targets.size();i++){
    const Matrix &m = targets.at(i);
    if(m.rows() != target_rows)
      throw std::invalid_argument(
        "learn_batch(): all targets samples must have the same number of rows. "
        "Sample index: " + std::to_string(i) +
        ", expected rows: " + std::to_string(target_rows) +
        ", actual rows: " + std::to_string(m.rows())
      );
  }

  for(int i = 0;i < inputs.size();i++){
    const Matrix &input = inputs.at(i);
    const Matrix &target = targets.at(i);

    if(input.cols() != target.cols())
      throw std::invalid_argument(
        "learn_batch: input/target cols mismatch at index " + std::to_string(i)
      );
  }
  #endif

  for(int i = 0;i < inputs.size();i++){
    backward_network(forward_network(inputs[i]) - targets[i],false);
    step_network(targets[i].cols());
  }
}

//1列の行列を受け取る
std::vector<Matrix> Trainer::make_batches(const std::vector<Matrix> &matrixs,const int batch_size){
  const int matrixs_rows = matrixs.at(0).rows();
  #ifndef NDEBUG
  for(int i = 0;i < matrixs.size();i++){
    const Matrix &m = matrixs.at(i);
    if(m.rows() != matrixs_rows)
      throw std::invalid_argument(
        "learn(): all input samples must have the same number of rows. "
        "Sample index: " + std::to_string(i) +
        ", expected rows: " + std::to_string(matrixs_rows) +
        ", actual rows: " + std::to_string(m.rows())
      );
    if(m.cols() != 1)
      throw std::invalid_argument(
        "learn(): matrixs sample must be a column vector (n x 1). Sample index: " + std::to_string(i)
      );
  }
  #endif

  std::vector<Matrix> batch;

  const int batch_length = matrixs.size() / batch_size;
  const int batch_end = matrixs.size() % batch_size;

  for(int i = 0;i <= batch_length;i++){
    int size = batch_size;
    if(i == batch_length){
      if(batch_end == 0) break;
      size = batch_end;
    }

    Matrix batch_matrix(matrixs_rows,size);

    int col = 0;

    for(int j = i * batch_size;j < i * batch_size + size;j++){
      for(int row = 0;row < matrixs_rows;row++){
        batch_matrix(row,col) = matrixs.at(j).at(row,0);
      }
      col++;
    }

    batch.push_back(batch_matrix);
  }

  return batch;
}

void Trainer::load_model(){
  std::ifstream load(modelPath);

  //ファイルサイズをチェック
  load.seekg(0, std::ios::end);
  if(load.tellg() == 0){
    std::cout << "Model file is empty. Skipping load." << std::endl;
    return; // 空ファイルなら何もしない
  }
  load.seekg(0, std::ios::beg);

  //ファイルがあるかどうか
  if(!load){
    std::cout << "No model file found, starting with empty network." << std::endl;
    return;
  }

  nlohmann::ordered_json j;

  load >> j;

  layers.clear();
  for(const auto &layer:j["layers"]){
    layers.push_back(std::move(LayerFactory::create(layer)));
  }
}

void Trainer::save_model() const{
  std::ofstream save(modelPath);
  save << to_json().dump(2);
}

void Trainer::make_dense_model(const std::vector<int> &a,const std::vector<Activation> &acts,int seed){
  layers.clear();

  std::mt19937 gen(seed);

  for(int i = 0;i < acts.size();i++){
    auto d = std::make_unique<DenseLayer>(a.at(i), a.at(i + 1));
    d->random_init(gen);
    d->act = &Acts::getAct(acts.at(i).name);

    layers.push_back(std::move(d));
  }
}

}//namespace cobalt_715::nn