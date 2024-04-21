//------------------------------------------------------
// Prog    : Denoiser.cu
// auteur original  : Mignotte Max
// portage sur GPU : Godbert Paul
// date    :
// version : 1.0
// langage : CUDA C
// labo    : DIRO
// note    :
//------------------------------------------------------

//------------------------------------------------
// FICHIERS INCLUS -------------------------------
//------------------------------------------------
#include "Fonctions.h"

//------------------------------------------------
// DEFINITIONS -----------------------------------
//------------------------------------------------
#define NAME_VISUALISER "display "
#define NAME_IMG_IN "../Images/lena512"
#define NAME_IMG_OUT "../Images/lena512_Restored"
#define NAME_IMG_DEG "../Images/lena512_Degraded"
//------------------------------------------------
// PROTOTYPE DE FONCTIONS  -----------------------
//------------------------------------------------
//>Main Function
void DctDenoise(float **, float **, float **, int, int, float);

//>Gestion
void copy_matrix(float **, float **, int, int);
void FilteringDCT_8x8_(float **, float, int, int, float **, float ***);
void FilteringDCT_8x8(float **, float, int, int, float **, float ***);
void ZigZagThreshold(float, float *, int);
void copy_matrix_on_device(float *, float **, int, int);
void copy_matrix_on_host(float **, float *, int , int);
void copy_matrix_1d_to_2d(float*,float**,int,int);

__global__ void HardThreshold(float, float *, int);
__global__ void denoise_image(float *, float *, int, int, int, int);
__global__ void denoise_block(float *, float, int, int, int, float *, float *, void (*)(float, float*, int));

#define SIGMA_NOISE 30
#define NB_ITERATIONS 1
#define THRESHOLD 90
#define OVERLAP 1
#define HARD_THRESHOLD 1

#define ZOOM 1
#define QUIT 0

//------------------------------------------------
//------------------------------------------------
// FONCTIONS  ------------------------------------
//------------------------------------------------
//------------------------------------------------
//----------------------------------------------------------
// IterDctDenoise
//----------------------------------------------------------

__global__ void simple_kernel(float *input, float *output, int length, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < length && idy < width) {
        int index = idy * width + idx;  // Corrected index calculation
        output[index] = input[index];   // Simply copy input to output
    }
}

void DctDenoise(float **DataDegraded, float *DataFiltered_d, float **Data, int lgth, int wdth, float Thresh)
{
    int k;
    int SizeWindow;
    char Name_img[NBCHAR];

    // Parameter
    SizeWindow = 8;

    // Info
    printf("\n   ---------------- ");
    printf("\n    IterDctDenoise ");
    printf("\n   ----------------");
    printf("\n  Length:Width [%d][%d]", lgth, wdth);
    printf("\n  -----------------------");
    printf("\n   >> SigmaNoise = [%d]", SIGMA_NOISE);
    printf("\n  -----------------------");
    printf("\n  Threshold_Dct  > [%.1d]", THRESHOLD);
    printf("\n  Size Window    > [%d]", SizeWindow);
    printf("\n  Overlap        > [%d]", OVERLAP);
    printf("\n\n");

    // Allocation Memoire
    float*** mat3d = fmatrix_allocate_3d(SizeWindow * SizeWindow, lgth, wdth);
    float*   DataFilteredDst_d = fmatrix_allocate_2d_device(lgth, wdth);
    float**  DataFiltered_h = fmatrix_allocate_2d(lgth, wdth);

    // Init
    copy_matrix_on_device(DataFiltered_d, DataDegraded, lgth, wdth);

    // Define block size
    int blockSize = SizeWindow; 

    // Calculate grid dimensions
    int blocksX = (lgth + blockSize - 1) / blockSize;
    int blocksY = (wdth + blockSize - 1) / blockSize;

    // Set up the thread block and grid dimensions
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(blocksX, blocksY);

    printf("Threads Per Block: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Blocks Per Grid: %d x %d\n", blocksPerGrid.x, blocksPerGrid.y);

    // Debug print before denoising
    // Launch kernel for denoising
    for (k = 0; k < NB_ITERATIONS; k++)
    {
        /*
        for(int torioidalShiftY = 0; torioidalShiftY < 8; torioidalShiftY++)
        {
            for(int torioidalShiftX = 0; torioidalShiftX < 8; torioidalShiftX++)
            {*/
                cudaError_t cudaStatus;

                denoise_image<<<blocksPerGrid, threadsPerBlock>>>(DataFiltered_d, DataFilteredDst_d, lgth, wdth, 0,0);
                cudaStatus = cudaDeviceSynchronize();

                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                    exit(1);
                }
                // Allocate host buffer for the current toroidal shift
                /*float* tempBuffer = (float*)malloc(lgth * wdth * sizeof(float));

                // Copy data from device to the host buffer
                cudaMemcpy(tempBuffer, DataFilteredDst_d, lgth * wdth * sizeof(float), cudaMemcpyDeviceToHost);

                // Calculate the offset for the current toroidal shift
                int offset = (torioidalShiftY * 8 + torioidalShiftX);

                // Copy data from the host buffer to the appropriate location in the 3D host array
                for (int i = 0; i < lgth; i++)
                {
                    for (int j = 0; j < wdth; j++)
                    {
                        mat3d[offset][i][j] = tempBuffer[i * wdth + j];
                    }
                }

                // Free the temporary host buffer
                free(tempBuffer);*/
  /*          }
        }*/
    }

                printf("oue\n");


    cudaMemcpy(DataFiltered_d, DataFilteredDst_d, lgth*wdth*sizeof(float), cudaMemcpyDeviceToDevice);
/*
    for (int i = 0; i < lgth; i++)
    {
        for (int j = 0; j < wdth; j++)
        {
            float temp = 0.0;
            double nb = 0.0;
            for (k = 0; k < 64; k++)
            {
                if (mat3d[k][i][j] > 0.0)
                {
                    nb++;
                    temp += mat3d[k][i][j];
                }
            }
            if (nb)
            {
                temp /= nb;
                DataFiltered_h[i][j] = temp;
            }
        }
    }
    printf("chef ?\n");

    copy_matrix_on_device(DataFiltered_d, DataFiltered_h, lgth, wdth);

    printf("chef ?2\n");
*/
    // Free memory
    if(DataFiltered_h)
        free_fmatrix_2d(DataFiltered_h);
    free_matrix_device(DataFilteredDst_d);
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

void copy_matrix_1d_to_2d(float *mat1, float **mat2, int lgth, int wdth)
{
    int i, j;

    for (i = 0; i < lgth; i++)
        for (j = 0; j < wdth; j++)
            mat2[i][j] = mat1[i * wdth + j];
}

void copy_matrix_2d_to_1d(float **mat1, float *mat2, int lgth, int wdth)
{
    int i, j;

    for (i = 0; i < lgth; i++)
        for (j = 0; j < wdth; j++)
            mat2[i * wdth + j] = mat1[i][j];
}

void copy_matrix_on_device(float *mat1, float **mat2, int lgth, int wdth)
{
    // Allocate memory for the temporary buffer on the heap
    float* buff = new float[lgth * wdth];

    for (int i = 0; i < lgth; i++)
        for (int j = 0; j < wdth; j++)
            buff[i * wdth + j] = mat2[i][j];

    // Calculate the size of the memory to copy
    size_t size = lgth * wdth * sizeof(float);

    // Copy the data from the host buffer (buff) to the device (mat1)
    cudaMemcpy(mat1, buff, size, cudaMemcpyHostToDevice);

    // Free the allocated memory for the host buffer
    cudaFree(buff);
}

void copy_matrix_on_host(float **mat1, float *mat2, int lgth, int wdth)
{
    float *buff = (float *)malloc(lgth * wdth * sizeof(float));

    // Copy matrix data from device to host buffer
    cudaMemcpy(buff, mat2, lgth * wdth * sizeof(float), cudaMemcpyDeviceToHost);

    // Unflatten the 1D buffer to 2D matrix
    for (int i = 0; i < lgth; i++)
        for (int j = 0; j < wdth; j++)
            mat1[i][j] = buff[wdth * i + j];

    // Free the temporary buffer
    free(buff);
}
//----------------------------------------------------------
// Fast FilteringDCT 8x8  <simple & optimise>
//----------------------------------------------------------
__global__ void denoise_image(float *Src, float *Dst, int ImgWidth, int ImgHeight, int toroidalShiftX, int toroidalShiftY) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int OffsetXBlocks = ((bx * 8 + toroidalShiftX) % 8);
    int OffsetYBlocks = ((by * 8 + toroidalShiftY) % 8);

    dim3 blockSize(8, 8);
    dim3 gridSize((ImgWidth + 7) / 8, (ImgHeight + 7) / 8);

    CUDA_DCT8x8<<<gridSize, blockSize,128>>>(Dst, ImgWidth, OffsetXBlocks, OffsetYBlocks, Src);

    __syncthreads();

    dim3 thresholdBlockSize(8, 8);
    dim3 thresholdGridSize(1, 1);
    
    HardThreshold<<<thresholdGridSize, thresholdBlockSize>>>(SIGMA_NOISE, Dst, 8);

    __syncthreads();

    //CUDA_IDCT8x8<<<gridSize, blockSize,128>>>(Dst, ImgWidth, OffsetXBlocks, OffsetYBlocks, Dst);

    //__syncthreads();

}

//----------------------------------------------------------
//----------------------------------------------------------
// Fast FilteringDCT 8x8  <simple> <ovl>
//----------------------------------------------------------
//----------------------------------------------------------
// void FilteringDCT_8x8_(float **imgin, float sigma, int length, int width, float **SquWin, float ***mat3d)
// {
//     int i, j;
//     int k, l;
//     int x, y;
//     int pos;
//     float temp;
//     float nb;
//     int posr, posc;
//     int overlap;

//     // Initialisation
//     //--------------
//     //>Record
//     overlap = OVERLAP;

//     //>Init
//     for (k = 0; k < 64; k++)
//         for (i = 0; i < length; i++)
//             for (j = 0; j < width; j++)
//                 mat3d[k][i][j] = -1.0;
//     // Loop
//     //----
//     for (i = 0; i < length; i += overlap)
//         for (j = 0; j < width; j += overlap)
//         {
//             for (k = 0; k < 8; k++)
//                 for (l = 0; l < 8; l++)
//                 {
//                     posr = i - 4 + k;
//                     posc = j - 4 + l;

//                     if (posr < 0)
//                         posr += length;
//                     if (posr > (length - 1))
//                         posr -= length;

//                     if (posc < 0)
//                         posc += width;
//                     if (posc > (width - 1))
//                         posc -= width;

//                     SquWin[k][l] = imgin[posr][posc];
//                 }

//             ddct8x8s(-1, SquWin);
//             if (HARD_THRESHOLD)
//                 HardThreshold(sigma, SquWin, 8);
//             if (!HARD_THRESHOLD)
//                 ZigZagThreshold(sigma, SquWin, 8);
//             ddct8x8s(1, SquWin);

//             x = (i % 8);
//             y = (j % 8);
//             pos = ((x * 8) + y);

//             for (k = 0; k < 8; k++)
//                 for (l = 0; l < 8; l++)
//                 {
//                     posr = i - 4 + k;
//                     posc = j - 4 + l;

//                     if (posr < 0)
//                         posr += length;
//                     if (posr > (length - 1))
//                         posr -= length;

//                     if (posc < 0)
//                         posc += width;
//                     if (posc > (width - 1))
//                         posc -= width;

//                     if (mat3d[pos][posr][posc] != -1)
//                         printf("!");

//                     mat3d[pos][posr][posc] = SquWin[k][l];
//                 }
//         }

//     // Averaging
//     //---------
//     for (i = 0; i < length; i++)
//         for (j = 0; j < width; j++)
//         {
//             temp = 0.0;
//             nb = 0.0;
//             for (k = 0; k < 64; k++)
//                 if (mat3d[k][i][j] > 0.0)
//                 {
//                     nb++;
//                     temp += mat3d[k][i][j];
//                 }

//             if (nb)
//             {
//                 temp /= nb;
//                 imgin[i][j] = temp;
//             }
//         }
// }
//----------------------------------------------------------
//  DCT thresholding
//----------------------------------------------------------
__global__ void HardThreshold(float sigma, float *coef, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        int index = i + N * j;

        if (fabs(coef[index]) < sigma)
        {
            coef[index] = 0.0;
        }
    }
}
//----------------------------------------------------------
//  DCT ZigZag thresholding
//----------------------------------------------------------
__device__ void ZigZagThreshold(float sigma, float *coef, int N)
{
    int result[8][8];
    int i = 0;
    int j = 0;
    int d = -1;
    int start = 0;
    int end = (N * N) - 1;

    //>ZigZag Matrix
    do
    {
        result[i][j] = start++;
        result[N - i - 1][N - j - 1] = end--;

        i += d;
        j -= d;
        if (i < 0)
        {
            i++;
            d = -d;
        }
        else if (j < 0)
        {
            j++;
            d = -d;
        }
    } while (start < end);
    if (start == end)
        result[i][j] = start;

    //>Seuillage
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            if (result[i][j] >= sigma)
                coef[i + N * j] = 0.0;
}

//---------------------------------------------------------
//---------------------------------------------------------
// PROGRAMME PRINCIPAL   ----------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
int main(int argc, char **argv)
{
    printf("-------------Current GPU--------------\n");

    cudaSetDevice(0);
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    printf("Device Name: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Grid Size: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Max Threads Dimension: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("--------------------------------------\n");

    int length, width;
    char BufSystVisuImg[NBCHAR];

    //>Lecture Image
    float **Img = LoadImagePgm(NAME_IMG_IN, &length, &width);

    //>CPU Memory Allocation
    float **ImgDegraded = fmatrix_allocate_2d(length, width);
    float **ImgDenoised = fmatrix_allocate_2d(length, width);

    //>GPU Memory Allocation
    float *ImgDenoised_d = fmatrix_allocate_2d_device(length, width);

    copy_matrix(ImgDegraded, Img, length, width);

    copy_matrix(ImgDegraded, Img, length, width);    

    add_gaussian_noise(ImgDegraded,length,width,SIGMA_NOISE*SIGMA_NOISE);

    printf("\n\n  Info Noise");
    printf("\n  ------------");
    printf("\n    > MSE = [%.2f]", computeMMSE(ImgDegraded, Img, length));

    //=========================================================
    //== PROG =================================================
    //=========================================================

    printf("bonjour!\n");

    DctDenoise(ImgDegraded, ImgDenoised_d, Img, length, width, THRESHOLD);

    printf("haha :)\n");

    copy_matrix_on_host(ImgDenoised, ImgDenoised_d, length, width);

    //---------------------------------------------
    // SAUVEGARDE
    // -------------------
    // L'image d�grad�e             > ImgDegraded
    // Le resultat du debruitage    > ImgFiltered
    //----------------------------------------------
    SaveImagePgm(NAME_IMG_DEG, ImgDegraded, length, width);
    SaveImagePgm(NAME_IMG_OUT, ImgDenoised, length, width);

    //>Visu Img
    strcpy(BufSystVisuImg, NAME_VISUALISER);
    strcat(BufSystVisuImg, NAME_IMG_IN);
    strcat(BufSystVisuImg, ".pgm&");
    printf("\n > %s", BufSystVisuImg);
    system(BufSystVisuImg);

    // Visu ImgDegraded
    strcpy(BufSystVisuImg, NAME_VISUALISER);
    strcat(BufSystVisuImg, NAME_IMG_DEG);
    strcat(BufSystVisuImg, ".pgm&");
    printf("\n > %s", BufSystVisuImg);
    system(BufSystVisuImg);

    // Visu ImgFiltered
    strcpy(BufSystVisuImg, NAME_VISUALISER);
    strcat(BufSystVisuImg, NAME_IMG_OUT);
    strcat(BufSystVisuImg, ".pgm&");
    printf("\n > %s", BufSystVisuImg);
    system(BufSystVisuImg);

    //--------------- End -------------------------------------
    //----------------------------------------------------------

    // Liberation memoire pour les matrices
    if (Img)
        free_fmatrix_2d(Img);
    if (ImgDegraded)
        free_fmatrix_2d(ImgDegraded);
    if (ImgDenoised)
        free_fmatrix_2d(ImgDenoised);
        free_matrix_device(ImgDenoised_d);

    printf("\n C'est fini... \n");
    return 0;
}
