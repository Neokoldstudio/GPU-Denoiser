//------------------------------------------------------
// module  : DCT-Denoise.h
// author  : Mignotte Max
// date    : 
// version : 1.0
// language: C++
// labo    : DIRO
// note    : 
//------------------------------------------------------


#ifndef DCT_DENOISE_H
#define DCT_DENOISE_H

//------------------------------------------------
// INCLUDED FILES --------------------------------
//------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <new>

#include "DCTFunction.h"

//------------------------------------------------
// PARAMETERS ------------------------------------
//------------------------------------------------

//-------------
// Question 2.2
// ------------
// NB_ITERATIONS  [1]
// THRESHOLD      [6]
// OVERLAP        [8]  
// HARD_THRESHOLD [0]
// MSE=157
// 

//-------------
// Question 2.3
// ------------
// NB_ITERATIONS  [1]
// THRESHOLD      [100]
// OVERLAP        [8] 
// HARD_THRESHOLD [1]
// MSE=140.28
// 

//-------------
// Question 2.4
// ------------
// NB_ITERATIONS  [1]
// THRESHOLD      [90]
// OVERLAP        [1] 
// HARD_THRESHOLD [1]
// MSE=65
//

//-------------
// Question X.X
// ------------
// NB_ITERATIONS  [5]
// THRESHOLD      [SigmaNoise=30]
// OVERLAP        [1] 
// HARD_THRESHOLD [1]
// MSE=58.06
//

#define NB_ITERATIONS   1
#define THRESHOLD     90.0    //SigmaNoise=30
#define OVERLAP         1     //1
#define HARD_THRESHOLD  1     //0>ZIGZAG

//------------------------------------------------
// CLASS DCTDenoise  ----------------------------
//------------------------------------------------
class DCTDenoise
 { 
  public: 

  //Matrix
  float**  DataDegraded;
  float**  DataFiltered;
  float**  DataWithoutNoise;
;
  float**  SquWinImg;

  //>Image
  int  length,width;

  //>Noise
  float SigmaNoise;
  float Threshold_Dct;
   
 //Functions
  public:  
  DCTDenoise(float**,int,int);
   ~DCTDenoise(); 

   void Options(float**,float,float);
  void IterDctDenoise();
};
//---------------------------------------------

#endif
