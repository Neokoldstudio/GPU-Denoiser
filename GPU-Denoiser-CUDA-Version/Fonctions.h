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
__device__ void ddct8x8s(int, float *);

//>Degradation
float gaussian_noise(float, float);
void add_gaussian_noise(float **, int, int, float);

__device__ float gaussian_noise(float, float, curandState *);
__global__ void add_gaussian_noise_kernel(float *, int, int, float, curandState);
void add_gaussian_noise_to_matrix(float *, int, int, float);

//>Mesure
float computeMMSE(float **, float **, int);
