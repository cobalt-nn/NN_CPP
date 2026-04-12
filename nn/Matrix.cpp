#include <immintrin.h>
#include <stdexcept>
#include "Matrix.h"

namespace cobalt_715::nn{

//btは転置済み行列
#ifdef __AVX__
//m256 SIMD命令
void Matrix::dot_tile_m256_Bt(const double *ad,const double *btd,double *cd,
  const int acol,const int btcol,const int ccol,
  const int ii,const int i_max_index,const int jj,const int j_max_index,const int kk,const int k_max_index,int &i_end,int &j_end){

  i_end = ii + ((i_max_index - ii) / 4) * 4;
  j_end = jj + ((j_max_index - jj) / 4) * 4;
  const int k_end = kk + ((k_max_index - kk) / 4) * 4;

  for(int i = ii;i < i_end;i += 4){
    const double *adptr1 = ad + i * acol;
    const double *adptr2 = ad + (i + 1) * acol;
    const double *adptr3 = ad + (i + 2) * acol;
    const double *adptr4 = ad + (i + 3) * acol;

     for(int j = jj;j < j_end;j += 4){
      double c11=0,c12=0,c13=0,c14=0;
      double c21=0,c22=0,c23=0,c24=0;
      double c31=0,c32=0,c33=0,c34=0;
      double c41=0,c42=0,c43=0,c44=0;

      const double *btdptr1 = btd + j * btcol;
      const double *btdptr2 = btd + (j + 1) * btcol;
      const double *btdptr3 = btd + (j + 2) * btcol;
      const double *btdptr4 = btd + (j + 3) * btcol;

      double *cdptr1 = cd + i * ccol + j;
      double *cdptr2 = cd + (i + 1) * ccol + j;
      double *cdptr3 = cd + (i + 2) * ccol + j;
      double *cdptr4 = cd + (i + 3) * ccol + j;

      int k = kk;

      __m256d vc11 = _mm256_setzero_pd();
      __m256d vc12 = _mm256_setzero_pd();
      __m256d vc13 = _mm256_setzero_pd();
      __m256d vc14 = _mm256_setzero_pd();

      __m256d vc21 = _mm256_setzero_pd();
      __m256d vc22 = _mm256_setzero_pd();
      __m256d vc23 = _mm256_setzero_pd();
      __m256d vc24 = _mm256_setzero_pd();

      __m256d vc31 = _mm256_setzero_pd();
      __m256d vc32 = _mm256_setzero_pd();
      __m256d vc33 = _mm256_setzero_pd();
      __m256d vc34 = _mm256_setzero_pd();

      __m256d vc41 = _mm256_setzero_pd();
      __m256d vc42 = _mm256_setzero_pd();
      __m256d vc43 = _mm256_setzero_pd();
      __m256d vc44 = _mm256_setzero_pd();

      for(;k < k_end;k += 4){
        __m256d va1 = _mm256_loadu_pd(adptr1 + k);
        __m256d va2 = _mm256_loadu_pd(adptr2 + k);
        __m256d va3 = _mm256_loadu_pd(adptr3 + k);
        __m256d va4 = _mm256_loadu_pd(adptr4 + k);

        __m256d vb1 = _mm256_loadu_pd(btdptr1 + k);
        __m256d vb2 = _mm256_loadu_pd(btdptr2 + k);
        __m256d vb3 = _mm256_loadu_pd(btdptr3 + k);
        __m256d vb4 = _mm256_loadu_pd(btdptr4 + k);

        vc11 = _mm256_fmadd_pd(va1,vb1,vc11);
        vc12 = _mm256_fmadd_pd(va1,vb2,vc12);
        vc13 = _mm256_fmadd_pd(va1,vb3,vc13);
        vc14 = _mm256_fmadd_pd(va1,vb4,vc14);

        vc21 = _mm256_fmadd_pd(va2,vb1,vc21);
        vc22 = _mm256_fmadd_pd(va2,vb2,vc22);
        vc23 = _mm256_fmadd_pd(va2,vb3,vc23);
        vc24 = _mm256_fmadd_pd(va2,vb4,vc24);

        vc31 = _mm256_fmadd_pd(va3,vb1,vc31);
        vc32 = _mm256_fmadd_pd(va3,vb2,vc32);
        vc33 = _mm256_fmadd_pd(va3,vb3,vc33);
        vc34 = _mm256_fmadd_pd(va3,vb4,vc34);

        vc41 = _mm256_fmadd_pd(va4,vb1,vc41);
        vc42 = _mm256_fmadd_pd(va4,vb2,vc42);
        vc43 = _mm256_fmadd_pd(va4,vb3,vc43);
        vc44 = _mm256_fmadd_pd(va4,vb4,vc44);
      }

      c11 = hsum256_pd(vc11); c12 = hsum256_pd(vc12); c13 = hsum256_pd(vc13); c14 = hsum256_pd(vc14);
      c21 = hsum256_pd(vc21); c22 = hsum256_pd(vc22); c23 = hsum256_pd(vc23); c24 = hsum256_pd(vc24);
      c31 = hsum256_pd(vc31); c32 = hsum256_pd(vc32); c33 = hsum256_pd(vc33); c34 = hsum256_pd(vc34);
      c41 = hsum256_pd(vc41); c42 = hsum256_pd(vc42); c43 = hsum256_pd(vc43); c44 = hsum256_pd(vc44);

      for(;k < k_max_index;k++){
        double a1 = adptr1[k];
        double a2 = adptr2[k];
        double a3 = adptr3[k];
        double a4 = adptr4[k];

        double b1 = btdptr1[k];
        double b2 = btdptr2[k];
        double b3 = btdptr3[k];
        double b4 = btdptr4[k];

        c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3; c14 += a1 * b4;
        c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3; c24 += a2 * b4;
        c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3; c34 += a3 * b4;
        c41 += a4 * b1; c42 += a4 * b2; c43 += a4 * b3; c44 += a4 * b4;
      }

      cdptr1[0] += c11; cdptr1[1] += c12; cdptr1[2] += c13; cdptr1[3] += c14;
      cdptr2[0] += c21; cdptr2[1] += c22; cdptr2[2] += c23; cdptr2[3] += c24;
      cdptr3[0] += c31; cdptr3[1] += c32; cdptr3[2] += c33; cdptr3[3] += c34;
      cdptr4[0] += c41; cdptr4[1] += c42; cdptr4[2] += c43; cdptr4[3] += c44;
    }
  }
}

#endif

//レジスタタイル
void Matrix::dot_tile_reg_Bt(const double *ad,const double *btd,double *cd,
  const int acol,const int btcol,const int ccol,
  const int ii,const int i_max_index,const int jj,const int j_max_index,const int kk,const int k_max_index,int &i_end,int &j_end){

  i_end = ii + ((i_max_index - ii) / 4) * 4;
  j_end = jj + ((j_max_index - jj) / 8) * 8;


  for(int i = ii;i < i_end;i += 4){
    const double *adptr1 = ad + i * acol;
    const double *adptr2 = ad + (i + 1) * acol;
    const double *adptr3 = ad + (i + 2) * acol;
    const double *adptr4 = ad + (i + 3) * acol;

    for(int j = jj;j < j_end;j += 8){
      double c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0,c18=0;
      double c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0,c28=0;
      double c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0,c38=0;
      double c41=0,c42=0,c43=0,c44=0,c45=0,c46=0,c47=0,c48=0;

      const double *btdptr1 = btd + j * btcol;
      const double *btdptr2 = btd + (j + 1) * btcol;
      const double *btdptr3 = btd + (j + 2) * btcol;
      const double *btdptr4 = btd + (j + 3) * btcol;
      const double *btdptr5 = btd + (j + 4) * btcol;
      const double *btdptr6 = btd + (j + 5) * btcol;
      const double *btdptr7 = btd + (j + 6) * btcol;
      const double *btdptr8 = btd + (j + 7) * btcol;

      double *cdptr1 = cd + i * ccol + j;
      double *cdptr2 = cd + (i + 1) * ccol + j;
      double *cdptr3 = cd + (i + 2) * ccol + j;
      double *cdptr4 = cd + (i + 3) * ccol + j;

      for(int k = kk;k < k_max_index;k++){
        double a1 = adptr1[k];
        double a2 = adptr2[k];
        double a3 = adptr3[k];
        double a4 = adptr4[k];

        double b1 = btdptr1[k];
        double b2 = btdptr2[k];
        double b3 = btdptr3[k];
        double b4 = btdptr4[k];
        double b5 = btdptr5[k];
        double b6 = btdptr6[k];
        double b7 = btdptr7[k];
        double b8 = btdptr8[k];

        c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3; c14 += a1 * b4; c15 += a1 * b5; c16 += a1 * b6; c17 += a1 * b7; c18 += a1 * b8;
        c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3; c24 += a2 * b4; c25 += a2 * b5; c26 += a2 * b6; c27 += a2 * b7; c28 += a2 * b8;
        c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3; c34 += a3 * b4; c35 += a3 * b5; c36 += a3 * b6; c37 += a3 * b7; c38 += a3 * b8;
        c41 += a4 * b1; c42 += a4 * b2; c43 += a4 * b3; c44 += a4 * b4; c45 += a4 * b5; c46 += a4 * b6; c47 += a4 * b7; c48 += a4 * b8;
      }

      cdptr1[0] += c11; cdptr1[1] += c12; cdptr1[2] += c13; cdptr1[3] += c14; cdptr1[4] += c15; cdptr1[5] += c16; cdptr1[6] += c17; cdptr1[7] += c18;
      cdptr2[0] += c21; cdptr2[1] += c22; cdptr2[2] += c23; cdptr2[3] += c24; cdptr2[4] += c25; cdptr2[5] += c26; cdptr2[6] += c27; cdptr2[7] += c28;
      cdptr3[0] += c31; cdptr3[1] += c32; cdptr3[2] += c33; cdptr3[3] += c34; cdptr3[4] += c35; cdptr3[5] += c36; cdptr3[6] += c37; cdptr3[7] += c38;
      cdptr4[0] += c41; cdptr4[1] += c42; cdptr4[2] += c43; cdptr4[3] += c44; cdptr4[4] += c45; cdptr4[5] += c46; cdptr4[6] += c47; cdptr4[7] += c48;
    }
  }
}

//ブロック行列積になっている
void Matrix::dot_Bt(const Matrix &a,const Matrix &bt,Matrix &c){
  const int arow = a.rows();
  const int acol = a.cols();
  const int btrow = bt.rows();
  const int btcol = bt.cols();
  const int crow = c.rows();
  const int ccol = c.cols();

  #ifndef NDEBUG
  if(acol != btcol) throw std::invalid_argument("Matrix::dot dimension mismatch");
  if(crow != arow || ccol != btrow) throw std::invalid_argument("Matrix::dot_Bt dimension mismatch: c.rows() != a.rows() or c.cols() != bt.rows()");
  #endif

  std::fill(c.data().begin(),c.data().end(),0.0);

  const int BS = 128;//ブロックサイズ

  const double *__restrict ad = a.data().data();
  const double *__restrict btd = bt.data().data();
  double *__restrict cd = c.data().data();

  #pragma omp parallel for collapse(2)
  for(int ii = 0;ii < crow;ii += BS){
    int i_max_index = std::min(crow,ii + BS);

    for(int jj = 0;jj < ccol;jj += BS){
      int j_max_index = std::min(ccol,jj + BS);

      for(int kk = 0;kk < acol;kk += BS){
        int k_max_index = std::min(acol,kk + BS);

        int i_end = 0;
        int j_end = 0;

        #ifdef __AVX__
          //行列のサイズが小さいとSIMDが逆に遅くなるため
          if(crow * ccol < 2048){
            dot_tile_reg_Bt(ad,btd,cd,acol,btcol,ccol,ii,i_max_index,jj,j_max_index,kk,k_max_index,i_end,j_end);
          }else{
            dot_tile_m256_Bt(ad,btd,cd,acol,btcol,ccol,ii,i_max_index,jj,j_max_index,kk,k_max_index,i_end,j_end);
          }
        #else
          dot_tile_reg_Bt(ad,btd,cd,acol,btcol,ccol,ii,i_max_index,jj,j_max_index,kk,k_max_index,i_end,j_end);
        #endif

        //行の残り
        for(int i = i_end;i < i_max_index;i++){
          double *iccol = cd + i * ccol;
          const double *iacol = ad + i * acol;
          for(int j = jj;j < j_end;j++){
            const double *jbtcol = btd + j * btcol;
            double cij = 0;
            for(int k = kk;k < k_max_index;k++){
              cij += iacol[k] * jbtcol[k];
            }
            iccol[j] += cij;
          }
        }

        //列の残り
        for(int i = ii;i < i_end;i++){
          double *iccol = cd + i * ccol;
          const double *iacol = ad + i * acol;
          for(int j = j_end;j < j_max_index;j++){
            const double *jbtcol = btd + j * btcol;
            double cij = 0;
            for(int k = kk;k < k_max_index;k++){
              cij += iacol[k] * jbtcol[k];
            }
            iccol[j] += cij;
          }
        }

        //右下の残り
        for(int i = i_end;i < i_max_index;i++){
          double *iccol = cd + i * ccol;
          const double *iacol = ad + i * acol;
          for(int j = j_end;j < j_max_index;j++){
            const double *jbtcol = btd + j * btcol;
            double cij = 0;
            for(int k = kk;k < k_max_index;k++){
              cij += iacol[k] * jbtcol[k];
            }
            iccol[j] += cij;
          }
        }

      }
    }
  }
}

}//namespace cobalt_715::nn