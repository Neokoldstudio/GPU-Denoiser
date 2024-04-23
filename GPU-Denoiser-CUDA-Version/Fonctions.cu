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

__constant__ float DCTv8matrixT[] =
{
  0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f,
  0.4903926402016152f, 0.4157348061512726f, 0.2777851165098011f, 0.0975451610080642f, -0.0975451610080641f, -0.2777851165098010f, -0.4157348061512727f, -0.4903926402016152f,
  0.4619397662556434f, 0.1913417161825449f, -0.1913417161825449f, -0.4619397662556434f, -0.4619397662556434f, -0.1913417161825452f, 0.1913417161825450f, 0.4619397662556433f,
  0.4157348061512726f, -0.0975451610080641f, -0.4903926402016152f, -0.2777851165098011f, 0.2777851165098009f, 0.4903926402016153f, 0.0975451610080640f, -0.4157348061512721f,
  0.3535533905932738f, -0.3535533905932737f, -0.3535533905932738f, 0.3535533905932737f, 0.3535533905932738f, -0.3535533905932733f, -0.3535533905932736f, 0.3535533905932733f,
  0.2777851165098011f, -0.4903926402016152f, 0.0975451610080642f, 0.4157348061512727f, -0.4157348061512726f, -0.0975451610080649f, 0.4903926402016152f, -0.2777851165098008f,
  0.1913417161825449f, -0.4619397662556434f, 0.4619397662556433f, -0.1913417161825450f, -0.1913417161825453f, 0.4619397662556437f, -0.4619397662556435f, 0.1913417161825431f,
  0.0975451610080642f, -0.2777851165098011f, 0.4157348061512727f, -0.4903926402016153f, 0.4903926402016152f, -0.4157348061512720f, 0.2777851165098022f, -0.0975451610080625f
};

__constant__ int BLOCK_SIZE = 8;
__constant__ int BLOCK_SIZE_LOG2 = 3;

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

__global__ void CUDA_DCT8x8(float *Dst, int ImgWidth, float *Src) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int global_index_x = (bx * 8) + tx;
    const int global_index_y = (by * 8) + ty;

    // Check boundary condition
    if (global_index_x >= ImgWidth || global_index_y >= ImgWidth) return;

    const int global_index = global_index_y * ImgWidth + global_index_x;
    const int local_index = (ty * 8) + tx;

    extern __shared__ float shared_memory[];

    float *CurBlockLocal1 = shared_memory;
    float *CurBlockLocal2 = &shared_memory[64];  // Assuming 2 blocks of 8x8 float

    CurBlockLocal1[local_index] = Src[global_index];

    __syncthreads();

    float curelem = 0.0f;
    int DCTv8matrixIndex = (ty * 8)+tx;
    int CurBlockLocal1Index = tx;
    #pragma unroll

    for (int i = 0; i < 8; i++) {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 1;
        CurBlockLocal1Index += 1;
    }

    CurBlockLocal2[(ty << 3) + tx] = curelem;

    __syncthreads();

    curelem = 0.0f;
    int CurBlockLocal2Index = (ty << 3) + tx;
    DCTv8matrixIndex = (tx << 3);
    #pragma unroll

    for (int i = 0; i < 8; i++) {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrixT[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 1;
    }

    CurBlockLocal1[(ty << 3) + tx] = curelem;

    __syncthreads();

    Dst[global_index] = CurBlockLocal1[local_index];
}

__global__ void CUDA_IDCT8x8(float *Dst, int ImgWidth, float *Src) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int global_index_x = (bx * 8) + tx;
    const int global_index_y = (by * 8) + ty;

    // Check boundary condition
    if (global_index_x >= ImgWidth || global_index_y >= ImgWidth) return;

    const int global_index = global_index_y * ImgWidth + global_index_x;
    const int local_index = (ty * 8) + tx;

    extern __shared__ float shared_memory[];

    float *CurBlockLocal1 = shared_memory;
    float *CurBlockLocal2 = &shared_memory[64];  // Assuming 2 blocks of 8x8 float

    CurBlockLocal1[local_index] = Src[global_index];

    __syncthreads();

    float curelem = 0.0f;
    int DCTv8matrixIndex = (ty * 8);
    int CurBlockLocal1Index = tx;
    #pragma unroll

    for (int i = 0; i < 8; i++) {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 8;
        CurBlockLocal1Index += 1;
    }

    CurBlockLocal2[(ty << 3) + tx] = curelem;

    __syncthreads();

    curelem = 0.0f;
    int CurBlockLocal2Index = (ty << 3) + tx;
    DCTv8matrixIndex = (tx << 3);
    #pragma unroll

    for (int i = 0; i < 8; i++) {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrixT[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 8;
    }

    CurBlockLocal1[(ty << 3) + tx] = curelem;

    __syncthreads();

    Dst[global_index] = CurBlockLocal1[local_index];
}

__global__ void ToroidalShift(float *Dst, float *Src, int lgth, int wdth, int shiftX, int shiftY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < lgth && y < wdth)
    {
        // Calculate toroidal shifted indices
        int shiftedX = (x + shiftX) % lgth;
        int shiftedY = (y + shiftY) % wdth;

        // Copy pixel value from the source to the destination
        Dst[shiftedY * lgth + shiftedX] = Src[y * lgth + x];
    }
}

//-------------------//
//--- DEGRADATION ---//
//-------------------//
//----------------------------------------------------------
//  Gaussian noisee
//----------------------------------------------------------
float gaussian_noise(float var, float mean)
{
    float noise, theta;

    // Noise generation
    noise = sqrt(-2 * var * log(1.0 - ((float)rand() / RAND_MAX)));
    theta = (float)rand() * 1.9175345E-4 - PI;
    noise = noise * cos(theta);
    noise += mean;
    if (noise > GREY_LEVEL)
        noise = GREY_LEVEL;
    if (noise < 0)
        noise = 0;
    return noise;
}

// ----------------------------------------------------------
//  Add Gaussian noise
// ----------------------------------------------------------
void add_gaussian_noise(float **mat, int lgth, int wdth, float var)
{
    int i, j;

    // Loop
    for (i = 0; i < lgth; i++)
        for (j = 0; j < wdth; j++)
            if (var != 0.0)
                mat[i][j] = gaussian_noise(var, mat[i][j]);
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


//NVIDIA SECRET PART >:)

// Used in forward and inverse DCT
#define C_a 1.387039845322148f  //!< a = (2^0.5) * cos(    pi / 16);
#define C_b 1.306562964876377f  //!< b = (2^0.5) * cos(    pi /  8);
#define C_c 1.175875602419359f  //!< c = (2^0.5) * cos(3 * pi / 16);
#define C_d 0.785694958387102f  //!< d = (2^0.5) * cos(5 * pi / 16);
#define C_e 0.541196100146197f  //!< e = (2^0.5) * cos(3 * pi /  8);
#define C_f 0.275899379282943f  //!< f = (2^0.5) * cos(7 * pi / 16);

/**
*  Normalization constant that is used in forward and inverse DCT
*/
#define C_norm 0.3535533905932737f  // 1 / (8^0.5)


/**
**************************************************************************
*  Performs in-place DCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void CUDAsubroutineInplaceDCTvector(float *Vect0, int Step) {
  float *Vect1 = Vect0 + Step;
  float *Vect2 = Vect1 + Step;
  float *Vect3 = Vect2 + Step;
  float *Vect4 = Vect3 + Step;
  float *Vect5 = Vect4 + Step;
  float *Vect6 = Vect5 + Step;
  float *Vect7 = Vect6 + Step;

  float X07P = (*Vect0) + (*Vect7);
  float X16P = (*Vect1) + (*Vect6);
  float X25P = (*Vect2) + (*Vect5);
  float X34P = (*Vect3) + (*Vect4);

  float X07M = (*Vect0) - (*Vect7);
  float X61M = (*Vect6) - (*Vect1);
  float X25M = (*Vect2) - (*Vect5);
  float X43M = (*Vect4) - (*Vect3);

  float X07P34PP = X07P + X34P;
  float X07P34PM = X07P - X34P;
  float X16P25PP = X16P + X25P;
  float X16P25PM = X16P - X25P;

  (*Vect0) = C_norm * (X07P34PP + X16P25PP);
  (*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
  (*Vect4) = C_norm * (X07P34PP - X16P25PP);
  (*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

  (*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
  (*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
  (*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
  (*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

/**
**************************************************************************
*  Performs in-place IDCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void CUDAsubroutineInplaceIDCTvector(float *Vect0, int Step) {
  float *Vect1 = Vect0 + Step;
  float *Vect2 = Vect1 + Step;
  float *Vect3 = Vect2 + Step;
  float *Vect4 = Vect3 + Step;
  float *Vect5 = Vect4 + Step;
  float *Vect6 = Vect5 + Step;
  float *Vect7 = Vect6 + Step;

  float Y04P = (*Vect0) + (*Vect4);
  float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y7f1aP3c5dPP =
      C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
  float Y7a1fM3d5cMP =
      C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

  float Y04M = (*Vect0) - (*Vect4);
  float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;
  float Y1c7dM3f5aPM =
      C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
  float Y1d7cP3a5fMM =
      C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

  (*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  (*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  (*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  (*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  (*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  (*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  (*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  (*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients. 2nd
*implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

__global__ void CUDAkernel2DCT(float *dst, int ImgStride,float *src) {

  __shared__ float block[8 * 8];

  int OffsThreadInRow = threadIdx.y * 64 + threadIdx.x;
  int OffsThreadInCol = threadIdx.z * 64;
  src += (blockIdx.y * 8 + OffsThreadInCol, ImgStride) +
         blockIdx.x * 8 + OffsThreadInRow;
  dst += (blockIdx.y * 8 + OffsThreadInCol, ImgStride) +
         blockIdx.x * 8 + OffsThreadInRow;
  float *bl_ptr =
      block + OffsThreadInCol * 8 + OffsThreadInRow;

#pragma unroll

  for (unsigned int i = 0; i < 64; i++)
    bl_ptr[i * 8] = src[i * ImgStride];

  __syncthreads();

  // process rows
  CUDAsubroutineInplaceDCTvector(
      block + (OffsThreadInCol + threadIdx.x) * 8 +
          OffsThreadInRow - threadIdx.x,
      1);

  __syncthreads();

  // process columns
  CUDAsubroutineInplaceDCTvector(bl_ptr, 8);

  __syncthreads();

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    dst[i * ImgStride] = bl_ptr[i * 8];
}

/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

__global__ void CUDAkernel2IDCT(float *dst, int ImgStride,float *src) {

  __shared__ float block[8 * 8];

  int OffsThreadInRow = threadIdx.y * 64 + threadIdx.x;
  int OffsThreadInCol = threadIdx.z * 64;
  src += (blockIdx.y * 8 + OffsThreadInCol, ImgStride) +
         blockIdx.x * 8 + OffsThreadInRow;
  dst += (blockIdx.y * 8 + OffsThreadInCol, ImgStride) +
         blockIdx.x * 8 + OffsThreadInRow;
  float *bl_ptr =
      block + OffsThreadInCol * 8 + OffsThreadInRow;

#pragma unroll

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    bl_ptr[i * 8] = src[i * ImgStride];

  __syncthreads();
  // process rows
  CUDAsubroutineInplaceIDCTvector(
      block + (OffsThreadInCol + threadIdx.x) * 8 +
          OffsThreadInRow - threadIdx.x,
      1);

  __syncthreads();
  // process columns
  CUDAsubroutineInplaceIDCTvector(bl_ptr, 8);

  __syncthreads();

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    dst[i * ImgStride] = bl_ptr[i * 8];
}