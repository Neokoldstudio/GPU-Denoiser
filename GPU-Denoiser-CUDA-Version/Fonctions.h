//------------------------------------------------------
// module  : Fonctions.h
// auteur original  : Mignotte Max
// portage sur GPU : Godbert Paul
// date    :
// version : 1.0
// langage : CUDA C
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
#include <curand_kernel.h>

//------------------------------------------------
// CONSTANTS & DEFINITIONS -----------------------
//------------------------------------------------
#define CARRE(X) ((X) * (X))
#define PI 3.1415926535897
#define GREY_LEVEL 255
#define NBCHAR 200

//------------------------------------------------
// PROTOTYPE -------------------------------------
//------------------------------------------------

//>Matrix
float *fmatrix_allocate_1d(int);
float **fmatrix_allocate_2d(int, int);
float ***fmatrix_allocate_3d(int, int, int);
float *fmatrix_allocate_1d_device(int);
float *fmatrix_allocate_2d_device(int, int);
float *fmatrix_allocate_3d_device(int, int, int);

void free_fmatrix_1d(float *);
void free_fmatrix_2d(float **);
void free_fmatrix_3d(float ***, int);
void free_matrix_device(float *);

//>Load/Save File
float **LoadImagePgm(char *, int *, int *);
void SaveImagePgm(char *, float **, int, int);

//>Fourier
__global__ void CUDA_DCT8x8(float *, int, float *);
__global__ void CUDA_IDCT8x8(float *, int, float *);
__global__ void CUDAkernel2DCT(float *, int,float *);
__global__ void CUDAkernel2IDCT(float *, int,float *);


//>Degradation
float gaussian_noise(float, float);
void add_gaussian_noise(float **, int, int, float);

//>Mesure
float computeMMSE(float **, float **, int);

//Image Processing
__global__ void ToroidalShift(float *, float *, int, int, int, int);
