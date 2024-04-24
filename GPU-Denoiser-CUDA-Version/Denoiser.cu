//------------------------------------------------------
// Prog    : Denoiser.cu
// auteur original  : Mignotte Max
// portage sur GPU : Godbert Paul
// version : 1.0
// langage : CUDA C
// labo    : DIRO
//------------------------------------------------------

//------------------------------------------------
// LIBRAIRIES ------------------------------------
//------------------------------------------------
#include <time.h>

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
void DctDenoise(float **, float *, int, int, float);

//>Gestion
void copy_matrix(float **, float **, int, int);
void FilteringDCT_8x8_(float **, float, int, int, float **, float ***);
void FilteringDCT_8x8(float **, float, int, int, float **, float ***);
void copy_matrix_on_device(float *, float **, int, int);
void copy_matrix_on_host(float **, float *, int, int);
void copy_matrix_1d_to_2d(float *, float **, int, int);
void copy_matrix_2d_to_1d(float **, float *, int, int);

__global__ void HardThreshold(float, float *, int);
__global__ void denoise_image(float *, float *, int, int, int, int);
__global__ void denoise_block(float *, float, int, int, int, float *, float *, void (*)(float, float *, int));

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

__global__ void simple_kernel(float *input, float *output, int length, int width)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < length && idy < width)
    {
        int index = idy * width + idx; // Corrected index calculation
        output[index] = input[index];  // Simply copy input to output
    }
}

void DctDenoise(float **DataDegraded, float *DataFiltered_d, int lgth, int wdth, float Thresh)
{
    int k;
    int SizeWindow;
    char Name_img[NBCHAR];

    // Parameter
    SizeWindow = 8;

    // Info
    printf("\n  --------------------- ");
    printf("\n      IterDctDenoise ");
    printf("\n  ---------------------");
    printf("\n      Length:Width [%d][%d]", lgth, wdth);
    printf("\n      -----------------------");
    printf("\n      >> SigmaNoise = [%d]", SIGMA_NOISE);
    printf("\n      -----------------------");
    printf("\n      Threshold_Dct  > [%.1d]", THRESHOLD);
    printf("\n      Size Window    > [%d]", SizeWindow);
    printf("\n      Overlap        > [%d]", OVERLAP);
    printf("\n\n");

    // Allocation Memoire
    float ***mat3d = fmatrix_allocate_3d(SizeWindow * SizeWindow, lgth, wdth);
    float *DataFilteredDst_d = fmatrix_allocate_2d_device(lgth, wdth);
    float **DataFiltered_h = fmatrix_allocate_2d(lgth, wdth);

    // Init
    copy_matrix_on_device(DataFiltered_d, DataDegraded, lgth, wdth);

    // Define block size
    int blockSize = SizeWindow;

    // Calculate grid dimensions
    int blocksX = (lgth + blockSize - 1) / blockSize;
    int blocksY = (wdth + blockSize - 1) / blockSize;

    // Set up the thread block and grid dimensions
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(lgth, wdth);

    printf("      --------Kernel Called on :------\n");
    printf("          Threads Per Block: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("          Blocks Per Grid: %d x %d\n", blocksPerGrid.x, blocksPerGrid.y);
    printf("      --------------------------------\n");

    // Debug print before denoising
    // Launch kernel for denoising
    for (k = 0; k < NB_ITERATIONS; k++)
    {
        for (int torioidalShiftY = 0; torioidalShiftY < 8; torioidalShiftY++)
        {
            for (int torioidalShiftX = 0; torioidalShiftX < 8; torioidalShiftX++)
            {
                // Toroidal Shift
                float *DataShifted_d;
                cudaMalloc((void **)&DataShifted_d, lgth * wdth * sizeof(float));

                int shiftX = torioidalShiftX;
                int shiftY = torioidalShiftY;

                ToroidalShift<<<blocksPerGrid, threadsPerBlock>>>(DataShifted_d, DataFiltered_d, lgth, wdth, shiftX, shiftY);
                cudaDeviceSynchronize();
                //  Launch Discrete Cosine Transform
                CUDA_DCT8x8<<<blocksPerGrid, threadsPerBlock>>>(DataFilteredDst_d, wdth, 0, 0, DataShifted_d);
                cudaDeviceSynchronize();

                // Launch Quantization kernel (here, a simple Hardthreshold)
                HardThreshold<<<blocksPerGrid, threadsPerBlock>>>(SIGMA_NOISE, DataFilteredDst_d, lgth);
                // CUDAkernelQuantizationFloat<<<blocksPerGrid, threadsPerBlock>>>(DataFilteredDst_d, lgth);
                cudaDeviceSynchronize();

                // Launch Inverse Discrete Cosine Transform
                CUDA_IDCT8x8<<<blocksPerGrid, threadsPerBlock>>>(DataFilteredDst_d, wdth, 0, 0, DataFilteredDst_d);
                cudaDeviceSynchronize();

                // Allocate host buffer for the current toroidal shift
                float *tempBuffer = (float *)malloc(lgth * wdth * sizeof(float));

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
                free(tempBuffer);
            }
        }
    }

    cudaMemcpy(DataFiltered_d, DataFilteredDst_d, lgth * wdth * sizeof(float), cudaMemcpyDeviceToDevice);

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

    copy_matrix_on_device(DataFiltered_d, DataFiltered_h, lgth, wdth);

    if (DataFiltered_h)
        free_fmatrix_2d(DataFiltered_h);
    free_matrix_device(DataFilteredDst_d);
}

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

//---------------------------------------------------------
//---------------------------------------------------------
// PROGRAMME PRINCIPAL   ----------------------------------
//---------------------------------------------------------
//---------------------------------------------------------

void usage(const char *programName)
{
    printf("Usage: %s [-3d] [<input_image.pgm>]\n", programName);
    printf("If -3d flag is provided, the program will process the image as a 3D image.\n");
}

void Denoise2D(char *inputImage)
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
    float **Img = LoadImagePgm(inputImage, &length, &width);

    if (!Img)
    {
        printf("Error loading image %s\n", inputImage);
        exit(1);
    }

    //>CPU Memory Allocation
    float **ImgDegraded = fmatrix_allocate_2d(length, width);
    float **ImgDenoised = fmatrix_allocate_2d(length, width);

    //>GPU Memory Allocation
    float *ImgDenoised_d = fmatrix_allocate_2d_device(length, width);

    copy_matrix(ImgDegraded, Img, length, width);
    add_gaussian_noise(ImgDegraded, length, width, SIGMA_NOISE * SIGMA_NOISE);

    printf("\n  Info Noise");
    printf("\n  ---------------------");
    printf("\n  Before Denoising :");
    printf("\n      > MSE = [%.2f]", computeMMSE(ImgDegraded, Img, length));

    clock_t start = clock();
    DctDenoise(ImgDegraded, ImgDenoised_d, length, width, THRESHOLD);
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    copy_matrix_on_host(ImgDenoised, ImgDenoised_d, length, width);

    printf("\n  ---------------------");
    printf("\n  After Denoising :");
    printf("\n      > MSE = [%.2f]", computeMMSE(ImgDenoised, Img, length));
    printf("\n  ---------------------");
    printf("\n  Duration :");
    printf("\n  Temps d'exécution de DctDenoise : %.2f ms\n", duration);
    printf("\n  ---------------------");

    SaveImagePgm(NAME_IMG_DEG, ImgDegraded, length, width);
    SaveImagePgm(NAME_IMG_OUT, ImgDenoised, length, width);

    strcpy(BufSystVisuImg, NAME_VISUALISER);
    strcat(BufSystVisuImg, inputImage);
    strcat(BufSystVisuImg, ".pgm&");
    printf("\n > %s", BufSystVisuImg);
    system(BufSystVisuImg);

    strcpy(BufSystVisuImg, NAME_VISUALISER);
    strcat(BufSystVisuImg, NAME_IMG_DEG);
    strcat(BufSystVisuImg, ".pgm&");
    printf("\n > %s", BufSystVisuImg);
    system(BufSystVisuImg);

    strcpy(BufSystVisuImg, NAME_VISUALISER);
    strcat(BufSystVisuImg, NAME_IMG_OUT);
    strcat(BufSystVisuImg, ".pgm&");
    printf("\n > %s", BufSystVisuImg);
    system(BufSystVisuImg);

    if (Img)
        free_fmatrix_2d(Img);
    if (ImgDegraded)
        free_fmatrix_2d(ImgDegraded);
    if (ImgDenoised)
        free_fmatrix_2d(ImgDenoised);
    free_matrix_device(ImgDenoised_d);
}

void Denoise3D(char *inputImage)
{
    int length, width, depth;
    float **Img;
    float ***Img3D;

    Img = LoadImagePgm(inputImage, &length, &width);

    //>CPU Memory Allocation
    float **ImgDegraded = fmatrix_allocate_2d(length, width);
    float **ImgDenoised = fmatrix_allocate_2d(length, width);

    // Allocate memory for 3D matrix and copy 2D image data
    Img3D = fmatrix_allocate_3d(length, length, width);
    for (int x = 0; x < length; ++x)
    {
        for (int y = 0; y < width; ++y)
        {
            Img3D[0][x][y] = Img[x][y];
        }
    }

    // Denoise along X-axis
    for (int x = 0; x < length; ++x)
    {
        float *ImgDenoised_d = fmatrix_allocate_2d_device(width, depth);

        float **slice = getSlice(Img3D, depth, length, width, x, 'x');
        DctDenoise(slice, ImgDenoised_d, width, depth, THRESHOLD);
        setSlice(Img3D, slice, depth, length, width, x, 'x');

        free_matrix_device(ImgDenoised_d);
    }

    // Denoise along Y-axis
    for (int y = 0; y < width; ++y)
    {
        float *ImgDenoised_d = fmatrix_allocate_2d_device(length, depth);

        float **slice = getSlice(Img3D, depth, length, width, y, 'y');
        DctDenoise(slice, ImgDenoised_d, length, depth, THRESHOLD);
        setSlice(Img3D, slice, depth, length, width, y, 'y');

        free_matrix_device(ImgDenoised_d);
    }

    // Denoise along Z-axis
    for (int z = 0; z < depth; ++z)
    {
        float *ImgDenoised_d = fmatrix_allocate_2d_device(length, width);

        float **slice = getSlice(Img3D, depth, length, width, z, 'z');
        DctDenoise(slice, ImgDenoised_d, length, width, THRESHOLD);
        setSlice(Img3D, slice, depth, length, width, z, 'z');

        free_matrix_device(ImgDenoised_d);
    }

    for (int x = 0; x < length; ++x)
    {
        for (int y = 0; y < width; ++y)
        {
            Img[x][y] = Img3D[0][x][y];
        }
    }

    SaveImagePgm(NAME_IMG_OUT, Img, length, width);

    // Free memory
    free_fmatrix_2d(Img);
    free_fmatrix_3d(Img3D, depth);
}

int main(int argc, char **argv)
{
    char *inputImage;
    int is3D = 0;
    if (argc == 1)
    {
        is3D = 0;
        inputImage = NAME_IMG_IN;
    }
    else if (argc == 2)
    {
        if (strcmp(argv[1], "-3d") == 0)
        {
            is3D = 1;
            inputImage = NAME_IMG_IN;
        }
        else
        {
            is3D = 0;
            inputImage = argv[1];
        }
    }
    else
    {
        usage(argv[0]);
        return 1;
    }

    if (is3D)
        Denoise3D(inputImage);
    else
        Denoise2D(inputImage);

    printf("\n C'est fini... \n");
    return 0;
}