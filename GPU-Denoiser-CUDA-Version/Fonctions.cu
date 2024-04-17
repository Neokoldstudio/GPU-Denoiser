//------------------------------------------------------
// module  : Fonctions.c
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
// INCLUDED FUNCTION -----------------------------
//------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

#include "Fonctions.h"

__constant__ float DCTv8matrix[] = {
  0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
  0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
  0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
  0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
  0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
  0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
  0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
  0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};



//--------------------------//
//-- Matrice de Flottant --//
//--------------------------//
//---------------------------------------------------------
//  Alloue de la memoire pour une matrice 1d de float
//---------------------------------------------------------
float *fmatrix_allocate_1d(int hsize)
{
    float *matrix;

    matrix = (float *)malloc(sizeof(float) * hsize);
    if (matrix == NULL)
    {
        printf("probleme d'allocation memoire");
        exit(-1);
    }

    return matrix;
}

float *fmatrix_allocate_1d_device(int hsize)
{
    float *matrix;
    // comme CUDA ne gère pas nativement les array multidimentionelles, on "applatis" la matrice 2D
    cudaMalloc((void **)&matrix, sizeof(float) * hsize);

    if (matrix == NULL)
    {
        printf("probleme d'allocation memoire dans une matrice 1d");
        exit(-1);
    }

    return matrix;
}

//----------------------------------------------------------
//  Alloue de la memoire pour une matrice 2d de float
//----------------------------------------------------------
float **fmatrix_allocate_2d(int vsize, int hsize)
{
    int i;
    float **matrix;
    float *imptr;

    matrix = (float **)malloc(sizeof(float *) * vsize);
    if (matrix == NULL)
    {
        printf("probleme d'allocation memoire");
        exit(-1);
    }

    imptr = (float *)malloc(sizeof(float) * hsize * vsize);
    if (imptr == NULL)
    {
        printf("probleme d'allocation memoire");
        exit(-1);
    }

    for (i = 0; i < vsize; i++, imptr += hsize)
        matrix[i] = imptr;
    return matrix;
}

float *fmatrix_allocate_2d_device(int vsize, int hsize)
{
    float *matrix;
    // comme CUDA ne gère pas nativement les array multidimentionelles, on "applatis" la matrice 2D
    cudaMalloc((void **)&matrix, sizeof(float) * vsize * hsize);

    if (matrix == NULL)
    {
        printf("probleme d'allocation memoire dans une matrice 2d");
        exit(-1);
    }

    return matrix;
}

//----------------------------------------------------------
// Allocation matrix 3d float
//----------------------------------------------------------
float ***fmatrix_allocate_3d(int dsize, int vsize, int hsize)
{
    int i;
    float ***matrix;

    matrix = (float ***)malloc(sizeof(float **) * dsize);

    for (i = 0; i < dsize; i++)
        matrix[i] = fmatrix_allocate_2d(vsize, hsize);
    return matrix;
}

float *fmatrix_allocate_3d_device(int dsize, int vsize, int hsize)
{
    float *matrix;
    // comme CUDA ne gère pas nativement les array multidimentionelles, on "applatis" la matrice 3D
    cudaMalloc((void **)&matrix, sizeof(float) * dsize * vsize * hsize);

    if (matrix == NULL)
    {
        printf("probleme d'allocation memoire dans une matrice 3d");
        exit(-1);
    }

    return matrix;
}

//----------------------------------------------------------
// Libere la memoire de la matrice 1d de float
//----------------------------------------------------------
void free_fmatrix_1d(float *pmat)
{
    free(pmat);
}

//----------------------------------------------------------
// Libere la memoire de la matrice 2d de float
//----------------------------------------------------------
void free_fmatrix_2d(float **pmat)
{
    free(pmat[0]);
    free(pmat);
}

//----------------------------------------------------------
// Free Memory 3d de float
//----------------------------------------------------------
void free_fmatrix_3d(float ***pmat, int dsize)
{
    int i;
    for (i = 0; i < dsize; i++)
    {
        free(pmat[i][0]);
        free(pmat[i]);
    }
    free(pmat);
}

//----------------------------------------------------------
// Free Device Memory of matrices
//----------------------------------------------------------
void free_matrix_device(float *pmat)
{
    cudaFree(pmat);
}

//--------------------//
//-- LOAD/SAVE/FILE --//
//--------------------//
//----------------------------------------------------------
// Chargement de l'image de nom <name> (en pgm)
//----------------------------------------------------------
float **LoadImagePgm(char *name, int *length, int *width)
{
    int i, j;
    unsigned char var;
    char buff[NBCHAR];
    int tmp;
    char *ptmp;
    float **mat;

    char stringTmp1[NBCHAR], stringTmp2[NBCHAR];

    int ta1, ta2, ta3;
    FILE *fic;

    //-----nom du fichier pgm-----
    strcpy(buff, name);
    strcat(buff, ".pgm");
    printf("---> Ouverture de %s", buff);

    //----ouverture du fichier----
    fic = fopen(buff, "r");
    if (fic == NULL)
    {
        printf("\n- Grave erreur a l'ouverture de %s  -\n", buff);
        exit(-1);
    }

    //--recuperation de l'entete--
    ptmp = fgets(stringTmp1, 100, fic);
    ptmp = fgets(stringTmp2, 100, fic);
    tmp = fscanf(fic, "%d %d", &ta1, &ta2);
    tmp = fscanf(fic, "%d\n", &ta3);

    //--affichage de l'entete--
    printf("\n\n--Entete--");
    printf("\n----------");
    printf("\n%s%s%d %d \n%d\n", stringTmp1, stringTmp2, ta1, ta2, ta3);

    *length = ta1;
    *width = ta2;
    mat = fmatrix_allocate_2d(*length, *width);

    //--chargement dans la matrice--
    for (i = 0; i < *length; i++)
        for (j = 0; j < *width; j++)
        {
            tmp = fread(&var, 1, 1, fic);
            mat[i][j] = var;
        }

    //---fermeture du fichier---
    fclose(fic);

    return (mat);
}

//----------------------------------------------------------
// Sauvegarde de l'image de nom <name> au format pgm
//----------------------------------------------------------
void SaveImagePgm(char *name, float **mat, int length, int width)
{
    int i, j;
    char buff[NBCHAR];
    FILE *fic;
    time_t tm;

    //--extension--
    strcpy(buff, name);
    strcat(buff, ".pgm");

    //--ouverture fichier--
    fic = fopen(buff, "w");
    if (fic == NULL)
    {
        printf(" Probleme dans la sauvegarde de %s", buff);
        exit(-1);
    }
    printf("\n Sauvegarde de %s au format pgm\n", name);

    //--sauvegarde de l'entete--
    fprintf(fic, "P5");
    if ((ctime(&tm)) == NULL)
        fprintf(fic, "\n#\n");
    else
        fprintf(fic, "\n# IMG Module, %s", ctime(&tm));
    fprintf(fic, "%d %d", width, length);
    fprintf(fic, "\n255\n");

    //--enregistrement--
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
            fprintf(fic, "%c", (char)mat[i][j]);

    //--fermeture fichier--
    fclose(fic);
}

//-------------//
//-- FOURIER --//
//-------------//
//----------------//
//-FAST DCT 8x8 --//
//----------------//
//----------------------//
// Made by T. OORA      //
//                      //
//----------------------//

/*
Short Discrete Cosine Transform
    data length :8x8, 16x16
    method      :row-column, radix 4 FFT
functions
    ddct8x8s  : 8x8 DCT
    ddct16x16s: 16x16 DCT
function prototypes
    void ddct8x8s(int isgn, double **a);
    void ddct16x16s(int isgn, double **a);
*/

/*
-------- 8x8 DCT (Discrete Cosine Transform) / Inverse of DCT --------
    [definition]
        <case1> Normalized 8x8 IDCT
            C[k1][k2] = (1/4) * sum_j1=0^7 sum_j2=0^7
                            a[j1][j2] * s[j1] * s[j2] *
                            cos(pi*j1*(k1+1/2)/8) *
                            cos(pi*j2*(k2+1/2)/8), 0<=k1<8, 0<=k2<8
                            (s[0] = 1/sqrt(2), s[j] = 1, j > 0)
        <case2> Normalized 8x8 DCT
            C[k1][k2] = (1/4) * s[k1] * s[k2] * sum_j1=0^7 sum_j2=0^7
                            a[j1][j2] *
                            cos(pi*(j1+1/2)*k1/8) *
                            cos(pi*(j2+1/2)*k2/8), 0<=k1<8, 0<=k2<8
                            (s[0] = 1/sqrt(2), s[j] = 1, j > 0)
    [usage]
        <case1>
            ddct8x8s(1, a);
        <case2>
            ddct8x8s(-1, a);
    [parameters]
        a[0...7][0...7] :input/output data (double **)
                         output data
                             a[k1][k2] = C[k1][k2], 0<=k1<8, 0<=k2<8
*/

/* Cn_kR = sqrt(2.0/n) * cos(pi/2*k/n) */
/* Cn_kI = sqrt(2.0/n) * sin(pi/2*k/n) */
/* Wn_kR = cos(pi/2*k/n) */
/* Wn_kI = sin(pi/2*k/n) */
#define C8_1R 0.49039264020161522456
#define C8_1I 0.09754516100806413392
#define C8_2R 0.46193976625564337806
#define C8_2I 0.19134171618254488586
#define C8_3R 0.41573480615127261854
#define C8_3I 0.27778511650980111237
#define C8_4R 0.35355339059327376220
#define W8_4R 0.70710678118654752440

__global__ void CUDA_DCT8x8(float *Dst, int ImgWidth, int OffsetXBlocks,
                               int OffsetYBlocks, float *Src) {
    // Block index
    const int bx = blockIdx.x + OffsetXBlocks;
    const int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current coefficient)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Calculate linear index for global memory and shared memory
    const int global_index = (by * 8 * ImgWidth) + (bx * 8) + (ty * ImgWidth) + tx;
    const int local_index = (ty * 8) + tx;

    extern __shared__ float shared_memory[];

    float *CurBlockLocal1 = shared_memory;
    float *CurBlockLocal2 = &shared_memory[64];  // 8 * 8 = 64

    // copy current image pixel to the first block
    CurBlockLocal1[local_index] = Src[global_index];

    // synchronize threads to make sure the block is copied
    __syncthreads();

    // calculate the multiplication of DCTv8matrixT * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = ty * 8;
    int CurBlockLocal1Index = 0 * 8 + tx;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 8;
        CurBlockLocal1Index += 8;
    }

    CurBlockLocal2[local_index] = curelem;

    // synchronize threads to make sure the first 2 matrices are multiplied and
    // the result is stored in the second block
    __syncthreads();

    // calculate the multiplication of (DCTv8matrixT * A) * DCTv8matrix and place
    // it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty * 8);
    DCTv8matrixIndex = 0 * 8 + tx;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 8;
    }

    CurBlockLocal1[local_index] = curelem;

    // synchronize threads to make sure the matrices are multiplied and the result
    // is stored back in the first block
    __syncthreads();

    // copy current coefficient to its place in the result array
    Dst[global_index] = CurBlockLocal1[local_index];
}

__global__ void CUDA_IDCT8x8(float *Dst, int ImgWidth, int OffsetXBlocks,
                                int OffsetYBlocks, float *TexSrc) {
  
    // Block index
    int bx = blockIdx.x + OffsetXBlocks;
    int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current image pixel)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory allocation for current block
    __shared__ float CurBlockLocal1[64];
    __shared__ float CurBlockLocal2[64];

    // Copy current image pixel to the shared memory
    int global_index = (by * 8 + ty) * ImgWidth + (bx * 8 + tx);
    CurBlockLocal1[ty * 8 + tx] = TexSrc[global_index];

    // Wait for all threads to complete copying
    __syncthreads();

    // Calculate the multiplication of DCTv8matrix * A and place it in the shared memory
    float curelem = 0;
    int DCTv8matrixIndex = ty * 8;
    int CurBlockLocal1Index = tx;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex++;
        CurBlockLocal1Index += 8;
    }

    CurBlockLocal2[ty * 8 + tx] = curelem;

    // Wait for all threads to complete the multiplication
    __syncthreads();

    // Calculate the multiplication of (DCTv8matrix * A) * DCTv8matrixT and place it in the shared memory
    curelem = 0;
    int CurBlockLocal2Index = ty * 8;
    DCTv8matrixIndex = tx;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index++;
        DCTv8matrixIndex += 8;
    }

    CurBlockLocal1[ty * 8 + tx] = curelem;

    // Wait for all threads to complete the multiplication
    __syncthreads();

    // Copy current coefficient to its place in the result array
    Dst[global_index] = CurBlockLocal1[ty * 8 + tx];
}
__device__ void ddct8x8s(int isgn, float *a)
{
    int j;
    float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
    float xr, xi;

    int M = 8; //block dimention 

    if (isgn < 0)
    {
        for (j = 0; j <= 7; j++)
        {
            x0r = a[0+ M * j] + a[7+ M * j];
            x1r = a[0+ M * j] - a[7+ M * j];
            x0i = a[2+ M * j] + a[5+ M * j];
            x1i = a[2+ M * j] - a[5+ M * j];
            x2r = a[4+ M * j] + a[3+ M * j];
            x3r = a[4+ M * j] - a[3+ M * j];
            x2i = a[6+ M * j] + a[1+ M * j];
            x3i = a[6+ M * j] - a[1+ M * j];
            xr = x0r + x2r;
            xi = x0i + x2i;
            a[0+ M * j] = C8_4R * (xr + xi);
            a[4+ M * j] = C8_4R * (xr - xi);
            xr = x0r - x2r;
            xi = x0i - x2i;
            a[2+ M * j] = C8_2R * xr - C8_2I * xi;
            a[6+ M * j] = C8_2R * xi + C8_2I * xr;
            xr = W8_4R * (x1i - x3i);
            x1i = W8_4R * (x1i + x3i);
            x3i = x1i - x3r;
            x1i += x3r;
            x3r = x1r - xr;
            x1r += xr;
            a[1+ M * j] = C8_1R * x1r - C8_1I * x1i;
            a[7+ M * j] = C8_1R * x1i + C8_1I * x1r;
            a[3+ M * j] = C8_3R * x3r - C8_3I * x3i;
            a[5+ M * j] = C8_3R * x3i + C8_3I * x3r;
        }
        for (j = 0; j <= 7; j++)
        {
            x0r = a[j+ M * 0] + a[j+ M * 7];
            x1r = a[j+ M * 0] - a[j+ M * 7];
            x0i = a[j+ M * 2] + a[j+ M * 5];
            x1i = a[j+ M * 2] - a[j+ M * 5];
            x2r = a[j+ M * 4] + a[j+ M * 3];
            x3r = a[j+ M * 4] - a[j+ M * 3];
            x2i = a[j+ M * 6] + a[j+ M * 1];
            x3i = a[j+ M * 6] - a[j+ M * 1];
            xr = x0r + x2r;
            xi = x0i + x2i;
            a[j+ M * 0] = C8_4R * (xr + xi);
            a[j+ M * 4] = C8_4R * (xr - xi);
            xr = x0r - x2r;
            xi = x0i - x2i;
            a[j+ M * 2] = C8_2R * xr - C8_2I * xi;
            a[j+ M * 6] = C8_2R * xi + C8_2I * xr;
            xr = W8_4R * (x1i - x3i);
            x1i = W8_4R * (x1i + x3i);
            x3i = x1i - x3r;
            x1i += x3r;
            x3r = x1r - xr;
            x1r += xr;
            a[j+ M * 1] = C8_1R * x1r - C8_1I * x1i;
            a[j+ M * 7] = C8_1R * x1i + C8_1I * x1r;
            a[j+ M * 3] = C8_3R * x3r - C8_3I * x3i;
            a[j+ M * 5] = C8_3R * x3i + C8_3I * x3r;
        }
    }
    else
    {
        for (j = 0; j <= 7; j++)
        {
            x1r = C8_1R * a[1+ M * j] + C8_1I * a[7+ M * j];
            x1i = C8_1R * a[7+ M * j] - C8_1I * a[1+ M * j];
            x3r = C8_3R * a[3+ M * j] + C8_3I * a[5+ M * j];
            x3i = C8_3R * a[5+ M * j] - C8_3I * a[3+ M * j];
            xr = x1r - x3r;
            xi = x1i + x3i;
            x1r += x3r;
            x3i -= x1i;
            x1i = W8_4R * (xr + xi);
            x3r = W8_4R * (xr - xi);
            xr = C8_2R * a[2+ M * j] + C8_2I * a[6+ M * j];
            xi = C8_2R * a[6+ M * j] - C8_2I * a[2+ M * j];
            x0r = C8_4R * (a[0+ M * j] + a[4+ M * j]);
            x0i = C8_4R * (a[0+ M * j] - a[4+ M * j]);
            x2r = x0r - xr;
            x2i = x0i - xi;
            x0r += xr;
            x0i += xi;
            a[0+ M * j] = x0r + x1r;
            a[7+ M * j] = x0r - x1r;
            a[2+ M * j] = x0i + x1i;
            a[5+ M * j] = x0i - x1i;
            a[4+ M * j] = x2r - x3i;
            a[3+ M * j] = x2r + x3i;
            a[6+ M * j] = x2i - x3r;
            a[1+ M * j] = x2i + x3r;
        }
        for (j = 0; j <= 7; j++)
        {
            x1r = C8_1R * a[j+ M * 1] + C8_1I * a[j+ M * 7];
            x1i = C8_1R * a[j+ M * 7] - C8_1I * a[j+ M * 1];
            x3r = C8_3R * a[j+ M * 3] + C8_3I * a[j+ M * 5];
            x3i = C8_3R * a[j+ M * 5] - C8_3I * a[j+ M * 3];
            xr = x1r - x3r;
            xi = x1i + x3i;
            x1r += x3r;
            x3i -= x1i;
            x1i = W8_4R * (xr + xi);
            x3r = W8_4R * (xr - xi);
            xr = C8_2R * a[j+ M * 2] + C8_2I * a[j+ M * 6];
            xi = C8_2R * a[j+ M * 6] - C8_2I * a[j+ M * 2];
            x0r = C8_4R * (a[j+ M * 0] + a[j+ M * 4]);
            x0i = C8_4R * (a[j+ M * 0] - a[j+ M * 4]);
            x2r = x0r - xr;
            x2i = x0i - xi;
            x0r += xr;
            x0i += xi;
            a[j+ M * 0] = x0r + x1r;
            a[j+ M * 7] = x0r - x1r;
            a[j+ M * 2] = x0i + x1i;
            a[j+ M * 5] = x0i - x1i;
            a[j+ M * 4] = x2r - x3i;
            a[j+ M * 3] = x2r + x3i;
            a[j+ M * 6] = x2i - x3r;
            a[j+ M * 1] = x2i + x3r;
        }
    }
}

//-------------------//
//--- DEGRADATION ---//
//-------------------//
//----------------------------------------------------------
//  Gaussian noisee
//----------------------------------------------------------
// float gaussian_noise(float var, float mean)
// {
//     float noise, theta;

//     // Noise generation
//     noise = sqrt(-2 * var * log(1.0 - ((float)rand() / RAND_MAX)));
//     theta = (float)rand() * 1.9175345E-4 - PI;
//     noise = noise * cos(theta);
//     noise += mean;
//     if (noise > GREY_LEVEL)
//         noise = GREY_LEVEL;
//     if (noise < 0)
//         noise = 0;
//     return noise;
// }

// ----------------------------------------------------------
//  Add Gaussian noise
// ----------------------------------------------------------
// void add_gaussian_noise(float **mat, int lgth, int wdth, float var)
// {
//     int i, j;

//     // Loop
//     for (i = 0; i < lgth; i++)
//         for (j = 0; j < wdth; j++)
//             if (var != 0.0)
//                 mat[i][j] = gaussian_noise(var, mat[i][j]);
// }

__device__ float gaussian_noise(float var, float mean, curandState *state)
{
    float noise, theta;

    // Noise generation
    noise = sqrtf(-2 * var * logf(1.0 - curand_uniform(state)));
    theta = curand_uniform(state) * 1.9175345E-4 - PI;
    noise = noise * cosf(theta);
    noise += mean;
    if (noise > GREY_LEVEL)
        noise = GREY_LEVEL;
    if (noise < 0)
        noise = 0;
    return noise;
}

__global__ void add_gaussian_noise_kernel(float *mat, int lgth, int wdth, float var, curandState *states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = x * wdth + y;

    // Each thread gets its own seed based on its global thread ID
    curand_init(index, 0, 0, &states[index]);

    if (x < lgth && y < wdth)
    {
        float noise = gaussian_noise(var, mat[index], &states[index]);
        mat[index] = noise;
    }
}

void add_gaussian_noise_to_matrix(float *cuMat, int width, int height, float var)
{
    // Set up CUDA random number generator states
    curandState *devStates;
    cudaMalloc((void **)&devStates, width * height * sizeof(curandState));

    // Launch kernel to add Gaussian noise to matrix
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    add_gaussian_noise_kernel<<<gridSize, blockSize>>>(cuMat, width, height, var, devStates);
    cudaDeviceSynchronize();

    // Free CUDA random number generator states
    cudaFree(devStates);
}

//--------------//
//--- MESURE ---//
//--------------//
//----------------------------------------------------------
// compute MMSE
//----------------------------------------------------------
float computeMMSE(float **mat1, float **mat2, int sz)
{
    int i, j;
    float mmse;

    // Boucle
    mmse = 0.0;
    for (i = 0; i < sz; i++)
        for (j = 0; j < sz; j++)
            mmse += CARRE(mat2[i][j] - mat1[i][j]);

    mmse /= (CARRE(sz));

    // retour
    return mmse;
}
