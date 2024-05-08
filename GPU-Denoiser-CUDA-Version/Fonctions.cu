//------------------------------------------------------
// module  : Fonctions.c
// auteur original  : Mignotte Max
// portage sur GPU : Godbert Paul
// version : 1.0
// langage : CUDA C
// labo    : DIRO
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

/**
 *  Wrapper to the fastest integer multiplication function on CUDA
 */
#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x, y) (__mul24(x, y))
#else
#define FMUL(x, y) ((x) * (y))
#endif

__constant__ float DCTv8matrix[] =
    {
        0.3535533905932738f, 0.4903926402016152f, 0.4619397662556434f, 0.4157348061512726f, 0.3535533905932738f, 0.2777851165098011f, 0.1913417161825449f, 0.0975451610080642f,
        0.3535533905932738f, 0.4157348061512726f, 0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
        0.3535533905932738f, 0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f, 0.0975451610080642f, 0.4619397662556433f, 0.4157348061512727f,
        0.3535533905932738f, 0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f, 0.3535533905932737f, 0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
        0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f, 0.2777851165098009f, 0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f, 0.4903926402016152f,
        0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f, 0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f, 0.4619397662556437f, -0.4157348061512720f,
        0.3535533905932738f, -0.4157348061512727f, 0.1913417161825450f, 0.0975451610080640f, -0.3535533905932736f, 0.4903926402016152f, -0.4619397662556435f, 0.2777851165098022f,
        0.3535533905932738f, -0.4903926402016152f, 0.4619397662556433f, -0.4157348061512721f, 0.3535533905932733f, -0.2777851165098008f, 0.1913417161825431f, -0.0975451610080625f};

__constant__ float DCTv8matrixT[] =
    {
        0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f, 0.3535533905932738f,
        0.4903926402016152f, 0.4157348061512726f, 0.2777851165098011f, 0.0975451610080642f, -0.0975451610080641f, -0.2777851165098010f, -0.4157348061512727f, -0.4903926402016152f,
        0.4619397662556434f, 0.1913417161825449f, -0.1913417161825449f, -0.4619397662556434f, -0.4619397662556434f, -0.1913417161825452f, 0.1913417161825450f, 0.4619397662556433f,
        0.4157348061512726f, -0.0975451610080641f, -0.4903926402016152f, -0.2777851165098011f, 0.2777851165098009f, 0.4903926402016153f, 0.0975451610080640f, -0.4157348061512721f,
        0.3535533905932738f, -0.3535533905932737f, -0.3535533905932738f, 0.3535533905932737f, 0.3535533905932738f, -0.3535533905932733f, -0.3535533905932736f, 0.3535533905932733f,
        0.2777851165098011f, -0.4903926402016152f, 0.0975451610080642f, 0.4157348061512727f, -0.4157348061512726f, -0.0975451610080649f, 0.4903926402016152f, -0.2777851165098008f,
        0.1913417161825449f, -0.4619397662556434f, 0.4619397662556433f, -0.1913417161825450f, -0.1913417161825453f, 0.4619397662556437f, -0.4619397662556435f, 0.1913417161825431f,
        0.0975451610080642f, -0.2777851165098011f, 0.4157348061512727f, -0.4903926402016153f, 0.4903926402016152f, -0.4157348061512720f, 0.2777851165098022f, -0.0975451610080625f};

// Temporary blocks
__shared__ float CurBlockLocal1[BLOCK_SIZE * BLOCK_SIZE];
__shared__ float CurBlockLocal2[BLOCK_SIZE * BLOCK_SIZE];

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
// Libere la memoire de la matrice 3d de float
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
// Libere la memoire de la matrice sur le device
//----------------------------------------------------------
void free_matrix_device(float *pmat)
{
    cudaFree(pmat);
}

//----------------------------------------------------------
// va chercher les slices d'une matrice 3D
//----------------------------------------------------------
void setSlice(float ***matrix3D, float **slice, int dim1, int dim2, int dim3, int index, char dimension)
{
    if (dimension == 'x')
    {
        slice = fmatrix_allocate_2d(dim2, dim3);
        for (int i = 0; i < dim2; ++i)
        {
            for (int j = 0; j < dim3; ++j)
            {
                matrix3D[index][i][j] = slice[i][j];
            }
        }
    }
    else if (dimension == 'y')
    {
        slice = fmatrix_allocate_2d(dim1, dim3);

        for (int i = 0; i < dim1; ++i)
        {
            for (int j = 0; j < dim3; ++j)
            {
                matrix3D[i][index][j] = slice[i][j];
            }
        }
    }
    else if (dimension == 'z')
    {
        slice = fmatrix_allocate_2d(dim1, dim2);

        for (int i = 0; i < dim1; ++i)
        {
            for (int j = 0; j < dim2; ++j)
            {
                matrix3D[i][j][index] = slice[i][j];
            }
        }
    }

    free_fmatrix_2d(slice);
}

float **getSlice(float ***matrix3D, int dim1, int dim2, int dim3, int index, char dimension)
{
    float **slice = NULL;

    if (dimension == 'x')
    {
        slice = fmatrix_allocate_2d(dim2, dim3);
        for (int i = 0; i < dim2; ++i)
        {
            for (int j = 0; j < dim3; ++j)
            {
                slice[i][j] = matrix3D[index][i][j];
            }
        }
    }
    else if (dimension == 'y')
    {
        slice = fmatrix_allocate_2d(dim1, dim3);

        for (int i = 0; i < dim1; ++i)
        {
            for (int j = 0; j < dim3; ++j)
            {
                slice[i][j] = matrix3D[i][index][j];
            }
        }
    }
    else if (dimension == 'z')
    {
        slice = fmatrix_allocate_2d(dim1, dim2);

        for (int i = 0; i < dim1; ++i)
        {
            for (int j = 0; j < dim2; ++j)
            {
                slice[i][j] = matrix3D[i][j][index];
            }
        }
    }

    return slice;
}

//---------------//
//--- GESTION ---//
//---------------//
//----------------------------------------------------------
// copy matrix
//----------------------------------------------------------
void copy_matrix(float **mat1, float **mat2, int lgth, int wdth)
{
    int i, j;

    for (i = 0; i < lgth; i++)
        for (j = 0; j < wdth; j++)
            mat1[i][j] = mat2[i][j];
}
//----------------------------------------------------------
// copy matrix 1d to 2d
//----------------------------------------------------------
void copy_matrix_1d_to_2d(float *mat1, float **mat2, int lgth, int wdth)
{
    int i, j;

    for (i = 0; i < lgth; i++)
        for (j = 0; j < wdth; j++)
            mat2[i][j] = mat1[i * wdth + j];
}
//----------------------------------------------------------
// copy matrix 2d to 1d
//----------------------------------------------------------
void copy_matrix_2d_to_1d(float **mat1, float *mat2, int lgth, int wdth)
{
    int i, j;

    for (i = 0; i < lgth; i++)
        for (j = 0; j < wdth; j++)
            mat2[i * wdth + j] = mat1[i][j];
}
//----------------------------------------------------------
// allocate an array on the GPU and copies the original array there
//----------------------------------------------------------
void copy_matrix_on_device(float *mat1, float **mat2, int lgth, int wdth)
{
    float *buff = new float[lgth * wdth];

    for (int i = 0; i < lgth; i++)
        for (int j = 0; j < wdth; j++)
            buff[i * wdth + j] = mat2[i][j];

    size_t size = lgth * wdth * sizeof(float);

    cudaMemcpy(mat1, buff, size, cudaMemcpyHostToDevice);

    cudaFree(buff);
}
//----------------------------------------------------------
// retrieve array from GPU to RAM
//----------------------------------------------------------
void copy_matrix_on_host(float **mat1, float *mat2, int lgth, int wdth)
{
    float *buff = (float *)malloc(lgth * wdth * sizeof(float));

    cudaMemcpy(buff, mat2, lgth * wdth * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < lgth; i++)
        for (int j = 0; j < wdth; j++)
            mat1[i][j] = buff[wdth * i + j];

    free(buff);
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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

__global__ void CUDA_DCT8x8(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks, float *Src)
{
    const int bx = blockIdx.x + OffsetXBlocks;
    const int by = blockIdx.y + OffsetYBlocks;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int global_index_x = (bx * BLOCK_SIZE) + tx;
    const int global_index_y = (by * BLOCK_SIZE) + ty;

    // Check boundary condition
    if (global_index_x >= ImgWidth || global_index_y >= ImgWidth)
        return;

    const int global_index = global_index_y * ImgWidth + global_index_x;
    const int local_index = (ty * BLOCK_SIZE) + tx;

    __shared__ float CurBlockLocal1[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float CurBlockLocal2[BLOCK_SIZE * BLOCK_SIZE];

    CurBlockLocal1[local_index] = Src[global_index] - 128.0f;

    __syncthreads();

    // calculate the multiplication of DCTv8matrixT * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = 0 * BLOCK_SIZE + ty;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += BLOCK_SIZE;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[local_index] = curelem;

    // synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    __syncthreads();

    // calculate the multiplication of (DCTv8matrixT * A) * DCTv8matrix and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += BLOCK_SIZE;
    }

    CurBlockLocal1[local_index] = curelem;

    // synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    __syncthreads();

    // copy current coefficient to its place in the result array
    Dst[((by << BLOCK_SIZE_LOG2) + ty) * ImgWidth + ((bx << BLOCK_SIZE_LOG2) + tx)] = CurBlockLocal1[local_index];
}

__global__ void CUDA_IDCT8x8(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks, float *Src)
{
    const int bx = blockIdx.x + OffsetXBlocks;
    const int by = blockIdx.y + OffsetYBlocks;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int global_index_x = (bx * BLOCK_SIZE) + tx;
    const int global_index_y = (by * BLOCK_SIZE) + ty;

    // Check boundary condition
    if (global_index_x >= ImgWidth || global_index_y >= ImgWidth)
        return;

    const int global_index = global_index_y * ImgWidth + global_index_x;
    const int local_index = (ty * BLOCK_SIZE) + tx;

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx] = Src[global_index];

    __syncthreads();

    // calculate the multiplication of DCTv8matrix * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = (ty << BLOCK_SIZE_LOG2) + 0;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 1;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx] = curelem;

    // synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    __syncthreads();

    // calculate the multiplication of (DCTv8matrix * A) * DCTv8matrixT and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = (tx << BLOCK_SIZE_LOG2) + 0;
#pragma unroll

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 1;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx] = curelem + 128.0f;

    // synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    __syncthreads();

    // copy current coefficient to its place in the result array
    Dst[FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx)] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx];
}

/**
**************************************************************************
*  Performs Image shifting
*
* \param shiftByBlocks  [IN] - [0:the image is shifted in its entierity / 1 : the image is shifted block by block]
*
* \return None
*/
__global__ void ToroidalShift(float *Dst, float *Src, int lgth, int wdth, int shiftX, int shiftY, int shiftByBlocks)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = (bx * BLOCK_SIZE) + tx;
    const int y = (by * BLOCK_SIZE) + ty;

    int shiftedX;
    int shiftedY;
    // compute the new index after shifting, modulo the dimentions of the image.
    if (shiftByBlocks)
    {
        shiftedX = (bx * BLOCK_SIZE) + (tx + shiftX) % wdth;
        shiftedY = (by * BLOCK_SIZE) + (ty + shiftY) % lgth;
    }
    else
    {
        shiftedX = (x + shiftX) % wdth;
        shiftedY = (y + shiftY) % lgth;
    }

    // then we update the destination image
    Dst[shiftedY * lgth + shiftedX] = Src[y * lgth + x];
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

    mmse = 0.0;
    for (i = 0; i < sz; i++)
        for (j = 0; j < sz; j++)
            mmse += CARRE(mat2[i][j] - mat1[i][j]);

    mmse /= (CARRE(sz));

    return mmse;
}

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// __constant__ short Q[] = {
//     32, 33, 51, 81, 66, 39, 34, 17, // this is Nvidia's quantization table. as you can see, the coefficient in the bottom right corner
//     33, 36, 48, 47, 28, 23, 12, 12, // are much lower than the ones in the upper left corner. This indicates that the higher frenquencies
//     51, 48, 47, 28, 23, 12, 12, 12, // will be prefered, which is not always what we aim for when using this technique for denoising.
//     81, 47, 28, 23, 12, 12, 12, 12, // I used this one because it gave me the less harsh results, but you can find and use another one if you like
//     66, 28, 23, 12, 12, 12, 12, 12,
//     39, 23, 12, 12, 12, 12, 12, 12,
//     34, 12, 12, 12, 12, 12, 12, 12,
//     17, 12, 12, 12, 12, 12, 12, 12};

__constant__ short Q[] = {
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1};

/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for floats plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationFloat(float *SrcDst, int Stride)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // copy current coefficient to the local variable
    float curCoef =
        SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)];
    float curQuant = (float)Q[ty * BLOCK_SIZE + tx];

    // quantize the current coefficient
    float quantized = roundf(curCoef / curQuant);
    curCoef = quantized * curQuant;

    // copy quantized coefficient back to the DCT-plane
    SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)] = curCoef;
}