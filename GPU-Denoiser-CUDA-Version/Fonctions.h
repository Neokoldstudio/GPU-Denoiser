//------------------------------------------------------
// module  : Fonctions.h
// auteur original  : Mignotte Max
// portage sur GPU : Godbert Paul
// version : 1.0
// langage : CUDA C
// labo    : DIRO
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
#define BLOCK_SIZE 8
#define BLOCK_SIZE_LOG2 3

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

void copy_matrix_on_device(float *, float **, int, int);
void copy_matrix_on_host(float **, float *, int, int);
void copy_matrix_1d_to_2d(float *, float **, int, int);
void copy_matrix_2d_to_1d(float **, float *, int, int);

float **getSlice(float ***, int, int, int, int, char);
void setSlice(float ***, float **, int, int, int, int, char);

//>Load/Save File
float **LoadImagePgm(char *, int *, int *);
void SaveImagePgm(char *, float **, int, int);

//>Fourier
__global__ void CUDA_DCT8x8(float *, int, int, int, float *);
__global__ void CUDA_IDCT8x8(float *, int, int, int, float *);
__global__ void CUDAkernel2DCT(float *, int, float *);
__global__ void CUDAkernel2IDCT(float *, int, float *);

__global__ void CUDAkernelQuantizationFloat(float *, int);

//>Degradation
float gaussian_noise(float, float);
void add_gaussian_noise(float **, int, int, float);

//>Mesure
float computeMMSE(float **, float **, int);

// Image Processing
__global__ void ToroidalShift(float *, float *, int, int, int, int);
