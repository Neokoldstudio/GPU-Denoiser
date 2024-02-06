//------------------------------------------------------
// module  : DCTFunction.h
// author : Mignotte Max
// date    :
// version : 1.0
// language: C++
// labo    : DIRO
// note    :
//------------------------------------------------------
// 

//------------------------------------------------
// INCLUDED FILES -------------------------------
//------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>

//------------------------------------------------
// CONSTANTS & DEFINITIONS -----------------------
//------------------------------------------------
#define CARRE(X) ((X)*(X))
#define PI  3.1415926535897


//------------------------------------------------
// PROTOTYPE -------------------------------------
//------------------------------------------------
float*    fmatrix_allocate_1d(int);
float**   fmatrix_allocate_2d(int,int);
float***  fmatrix_allocate_3d(int,int,int);
void      free_fmatrix_1d(float*);
void      free_fmatrix_2d(float**);
void      free_fmatrix_3d(float***,int);

void copy_matrix(float**,float**,int,int);

float computeMMSE(float**,float**,int);
void  computeMoy(float**,int);

void GetLengthWidth(char*,int*,int*);
void load_image(float**,char*,int,int);
void save_picture_pgm(char*,char*,float**,int,int);

void FilteringDCT_8x8_(float**,float,int,int,float**,float***);
void FilteringDCT_8x8(float**,float,int,int,float**,float***);

void ddct8x8s(int,float**);
void HardThreshold(float,float**,int);
void ZigZagThreshold(float,float**,int);

float gaussian_noise(float,float);
void  add_gaussian_noise(float**,int,int,float);


