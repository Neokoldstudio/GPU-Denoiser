//------------------------------------------------------
// module  : FunctionDemo5.h
// auteur  : Mignotte Max
// date    :
// version : 1.0
// langage : C++
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

//------------------------------------------------
// CONSTANTS & DEFINITIONS -----------------------
//------------------------------------------------
#define CARRE(X) ((X)*(X))
#define PI  3.1415926535897
#define GREY_LEVEL 255
#define NBCHAR 200

//------------------------------------------------
// PROTOTYPE -------------------------------------
//------------------------------------------------

//>Matrix
float*    fmatrix_allocate_1d(int);
float**   fmatrix_allocate_2d(int,int);
float***  fmatrix_allocate_3d(int,int,int);
void      free_fmatrix_1d(float*);
void      free_fmatrix_2d(float**);
void      free_fmatrix_3d(float***,int);

//>Load/Save File
float** LoadImagePgm(char*,int*,int*);
void SaveImagePgm(char*,float**,int,int);

//>Fourier
void ddct8x8s(int,float**);

//>Degradation
float gaussian_noise(float,float);
void  add_gaussian_noise(float**,int,int,float);

//>Mesure
float computeMMSE(float**,float**,int);





