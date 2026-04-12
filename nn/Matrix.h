#ifndef MATRIX_H
#define MATRIX_H
//#define NDEBUG

#include <vector>
#include <string>
#include <immintrin.h>

namespace cobalt_715::nn{

//行列クラス基本的な演算を用意している
class Matrix{
private:
  std::vector<double> data_;//一次元配列を[row * cols_ + col]でアクセスすることで実行速度を上げている
  int rows_;//行
  int cols_;//列

  //アクセス範囲を確認している
  //at()から呼び出している
  void check_index(int row,int col) const{
    #ifndef NDEBUG
    if(row < 0 || col < 0 || rows_ <= row || cols_ <= col) throw std::out_of_range("Matrix::at index out of range");
    #endif
  }

public:
  //要素を取り出す
  //列、行の順
  double& at(int row,int col){
    check_index(row,col);
    return data_[row * cols_ + col];
  }

  const double& at(int row,int col) const{
    check_index(row,col);
    return data_[row * cols_ + col];
  }

  double& operator()(int row,int col){
    return at(row,col);
  }

  const double& operator()(int row,int col) const{
    return at(row,col);
  }

  int rows() const{return rows_;}
  int cols() const{return cols_;}

  //参照を返しているため気を付けること
  std::vector<double>& data(){return data_;};
  const std::vector<double>& data() const{return data_;}

  //最適化のためbtは転置済み行列を受け取る
  //掛けられる行列、掛ける行列、代入される行列の順に受け取る。c = a * bになる
  //行列積もMNIST1エポック3.5秒ほどで学習できるぐらいには最適化している
  //ブロック行列積のブロックの部分からレジスタタイル、SIMD命令に派生する
  static void dot_Bt(const Matrix &a,const Matrix &bt,Matrix &c);

  //以下ブロック行列積の各ブロックから呼び出されている
  #ifdef __AVX__
  //m256 SIMD命令
  static void dot_tile_m256_Bt(const double *ad,const double *btd,double *cd,
    const int acol,const int btcol,const int ccol,
    const int ii,const int i_max_index,const int jj,const int j_max_index,const int kk,const int k_max_index,int &i_end,int &j_end);

  //__m256dの合計を返す
  inline static double hsum256_pd(__m256d v){
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(sum);
  }

  #endif

  //レジスタタイル
  static void dot_tile_reg_Bt(const double *ad,const double *btd,double *cd,
    const int acol,const int btcol,const int ccol,
    const int ii,const int i_max_index,const int jj,const int j_max_index,const int kk,const int k_max_index,int &i_end,int &j_end);

  Matrix operator*(const Matrix &rhs) const{
    Matrix c(rows_,rhs.cols());
    Matrix::dot_Bt(*this,rhs.T(),c);
    return c;
  }

  //スカラー倍　a * dの結果をoutに入れる
  static void scale(const Matrix &a,double d,Matrix &out){
    #ifndef NDEBUG
    if(a.rows() != out.rows() || a.cols() != out.cols()) throw std::invalid_argument("Matrix::add dimension mismatch");
    #endif

    const double *ad = a.data().data();
    double *od = out.data().data();

    const int size = a.rows() * a.cols();

    for(int i = 0;i < size;i++){
      od[i] = ad[i] * d;
    }
  }

  Matrix operator*(double d) const{
    Matrix m(rows_,cols_);
    scale(*this,d,m);
    return m;
  }

  //a + bの結果をoutに入れる
  static void add(const Matrix &a,const Matrix &b,Matrix &out){
    #ifndef NDEBUG
    if(a.rows() != b.rows() || a.cols() != b.cols() || a.rows() != out.rows() || a.cols() != out.cols())
      throw std::invalid_argument("Matrix::add dimension mismatch");
    #endif

    const double *ad = a.data().data();
    const double *bd = b.data().data();
    double *od = out.data().data();

    const int size = a.rows() * a.cols();

    for(int i = 0;i < size;i++){
      od[i] = ad[i] + bd[i];
    }
  }

  Matrix operator+(const Matrix &rhs) const{
    Matrix m(rows_,cols_);
    add(*this,rhs,m);
    return m;
  }

  Matrix& operator+=(const Matrix &rhs){
    add(rhs,*this,*this);
    return *this;
  }

  //a - bの結果をoutに入れる
  static void subtract(const Matrix &a,const Matrix &b,Matrix &out){
    #ifndef NDEBUG
    if(a.rows() != b.rows() || a.cols() != b.cols() || a.rows() != out.rows() || a.cols() != out.cols())
      throw std::invalid_argument("Matrix::add dimension mismatch");
    #endif

    const double *ad = a.data().data();
    const double *bd = b.data().data();
    double *od = out.data().data();

    const int size = a.rows() * a.cols();

    for(int i = 0;i < size;i++){
      od[i] = ad[i] - bd[i];
    }
  }

  Matrix operator-(const Matrix &rhs) const{
    Matrix m(rows_,cols_);
    subtract(*this,rhs,m);
    return m;
  }

  Matrix& operator-=(const Matrix &rhs){
    subtract(*this,rhs,*this);
    return *this;
  }

  //転置を返す
  Matrix T() const{
    Matrix t(cols_,rows_);

    double *td = t.data().data();
    const double *d = data_.data();

    for(int i = 0;i < rows_;i++){
      const int icols = i * cols_;
      for(int j = 0;j < cols_;j++){
        td[j * rows_ + i] = d[icols + j];
      }
    }
    return t;
  }

  //tに転置を保存する
  void T_to(Matrix &t) const{
    #ifndef NDEBUG
    if(rows_ != t.cols() || cols_ != t.rows()) throw std::invalid_argument("Matrix transpose dimension mismatch");
    #endif

    const double *d = data_.data();
    double *td = t.data().data();

    for(int i = 0;i < rows_;i++){
      const int icol = i * cols_;
      for(int j = 0;j < cols_;j++){
        td[j * rows_ + i] = d[icol + j];
      }
    }
  }

  //列、行の順にサイズを受け取る
  //initの数値で全要素初期化
  Matrix(int rows,int cols,double init=0) : rows_(rows),cols_(cols),data_(rows * cols,init){
    #ifndef NDEBUG
    if(rows <= 0 || cols <= 0) throw std::invalid_argument("Matrix size must be positive");
    #endif
  }

  //[row * cols_ + col]でアクセスされるため1行目 + ２行目のように書く
  Matrix(int rows,int cols,const std::vector<double> &data) : rows_(rows),cols_(cols),data_(data){
    #ifndef NDEBUG
    if(data_.size() != rows_ * cols_) throw std::invalid_argument("Matrix::data size mismatch");
    #endif
  }

  //2次元配列がそのまま行列になる
  Matrix(int rows,int cols,const std::vector<std::vector<double>> &data) : Matrix(rows,cols){
    #ifndef NDEBUG
    if(data.size() != rows_) throw std::invalid_argument("Matrix::data size mismatch");
    for(int i = 0;i < rows_;i++){
      if(data.at(i).size() != cols_) throw std::invalid_argument("Matrix::data size mismatch");
    }
    #endif
    for(int i = 0;i < rows_;i++){
      for(int j = 0;j < cols_;j++){
        data_[i * cols_ + j] = data[i][j];
      }
    }
  }

  std::string to_string() const{
    std::string s = "// " + std::to_string(rows_) + " * " + std::to_string(cols_);
    s += "\n{\n";
    for(int i = 0;i < rows_;i++){
      s += "{ ";
      for(int j = 0;j < cols_ - 1;j++){
        s += std::to_string(at(i,j)) + " , ";
      }
      s += std::to_string(at(i,cols_ - 1)) + " }";
      if(i != rows_ -1) s += " ,";
      s += "\n";
    }
    s += "}";
    return s;
  }
};

}//namespace cobalt_715::nn

#endif