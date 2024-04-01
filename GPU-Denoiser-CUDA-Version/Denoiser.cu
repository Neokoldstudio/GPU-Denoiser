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
void HardThreshold(float, float **, int);
void ZigZagThreshold(float, float **, int);

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
void DctDenoise(float **DataDegraded, float **DataFiltered, float **Data, int lgth, int wdth, float Thresh)
{
    int k;
    int SizeWindow;
    char Name_img[NBCHAR];

    // Parameter
    SizeWindow = 8;

    // Info
    //----;
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
    float **SquWin = fmatrix_allocate_2d(SizeWindow, SizeWindow);
    float ***mat3d = fmatrix_allocate_3d(SizeWindow * SizeWindow, lgth, wdth);

    // Init
    copy_matrix(DataFiltered, DataDegraded, lgth, wdth);

    //>Loop-DEnoising
    for (k = 0; k < NB_ITERATIONS; k++)
    {
        FilteringDCT_8x8_(DataFiltered, THRESHOLD, lgth, wdth, SquWin, mat3d);
        printf("\n   > MSE >> [%.5f]", computeMMSE(DataFiltered, Data, lgth));
    }

    // Free memory
    if (SquWin)
        free_fmatrix_2d(SquWin);
    if (mat3d)
        free_fmatrix_3d(mat3d, SizeWindow * SizeWindow);
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
    float buff[lgth * wdth];

    for (int i = 0; i < lgth; i++)
        for (int j = 0; j < wdth; j++)
            buff[wdth * i + j] = mat2[i][j];

    cudaMemcpy(mat2, buff, lgth * wdth, cudaMemcpyHostToDevice);
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
void FilteringDCT_8x8(float **imgin, float sigma, int length, int width, float **SquWin, float ***mat3d)
{
    int i, j, k, l;
    int x, y;
    int pos, posr, posc;
    float temp;
    float nb;

    // Boucle
    //------
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
        {
            for (k = 0; k < 8; k++)
                for (l = 0; l < 8; l++)
                {
                    posr = i - 4 + k;
                    posc = j - 4 + l;

                    if (posr < 0)
                        posr += length;
                    if (posr > (length - 1))
                        posr -= length;
                    if (posc < 0)
                        posc += width;
                    if (posc > (width - 1))
                        posc -= width;

                    SquWin[k][l] = imgin[posr][posc];
                }

            ddct8x8s(-1, SquWin);
            HardThreshold(sigma, SquWin, 8);
            ddct8x8s(1, SquWin);

            x = (i % 8);
            y = (j % 8);
            pos = ((x * 8) + y);

            for (k = 0; k < 8; k++)
                for (l = 0; l < 8; l++)
                {
                    posr = i - 4 + k;
                    posc = j - 4 + l;

                    if (posr < 0)
                        posr += length;
                    if (posr > (length - 1))
                        posr -= length;
                    if (posc < 0)
                        posc += width;
                    if (posc > (width - 1))
                        posc -= width;

                    mat3d[pos][posr][posc] = SquWin[k][l];
                }
        }

    // Moyennage
    //---------
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
        {
            temp = 0.0;
            nb = 0.0;
            for (k = 0; k < 64; k++)
                if (mat3d[k][i][j] > 0)
                {
                    nb++;
                    temp += mat3d[k][i][j];
                }

            if (nb)
            {
                temp /= nb;
                imgin[i][j] = temp;
            }
        }
}

//----------------------------------------------------------
//----------------------------------------------------------
// Fast FilteringDCT 8x8  <simple> <ovl>
//----------------------------------------------------------
//----------------------------------------------------------
void FilteringDCT_8x8_(float **imgin, float sigma, int length, int width, float **SquWin, float ***mat3d)
{
    int i, j;
    int k, l;
    int x, y;
    int pos;
    float temp;
    float nb;
    int posr, posc;
    int overlap;

    // Initialisation
    //--------------
    //>Record
    overlap = OVERLAP;

    //>Init
    for (k = 0; k < 64; k++)
        for (i = 0; i < length; i++)
            for (j = 0; j < width; j++)
                mat3d[k][i][j] = -1.0;

    // Loop
    //----
    for (i = 0; i < length; i += overlap)
        for (j = 0; j < width; j += overlap)
        {
            for (k = 0; k < 8; k++)
                for (l = 0; l < 8; l++)
                {
                    posr = i - 4 + k;
                    posc = j - 4 + l;

                    if (posr < 0)
                        posr += length;
                    if (posr > (length - 1))
                        posr -= length;

                    if (posc < 0)
                        posc += width;
                    if (posc > (width - 1))
                        posc -= width;

                    SquWin[k][l] = imgin[posr][posc];
                }

            ddct8x8s(-1, SquWin);
            if (HARD_THRESHOLD)
                HardThreshold(sigma, SquWin, 8);
            if (!HARD_THRESHOLD)
                ZigZagThreshold(sigma, SquWin, 8);
            ddct8x8s(1, SquWin);

            x = (i % 8);
            y = (j % 8);
            pos = ((x * 8) + y);

            for (k = 0; k < 8; k++)
                for (l = 0; l < 8; l++)
                {
                    posr = i - 4 + k;
                    posc = j - 4 + l;

                    if (posr < 0)
                        posr += length;
                    if (posr > (length - 1))
                        posr -= length;

                    if (posc < 0)
                        posc += width;
                    if (posc > (width - 1))
                        posc -= width;

                    if (mat3d[pos][posr][posc] != -1)
                        printf("!");

                    mat3d[pos][posr][posc] = SquWin[k][l];
                }
        }

    // Averaging
    //---------
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
        {
            temp = 0.0;
            nb = 0.0;
            for (k = 0; k < 64; k++)
                if (mat3d[k][i][j] > 0.0)
                {
                    nb++;
                    temp += mat3d[k][i][j];
                }

            if (nb)
            {
                temp /= nb;
                imgin[i][j] = temp;
            }
        }
}

//----------------------------------------------------------
//  DCT thresholding
//----------------------------------------------------------
void HardThreshold(float sigma, float **coef, int N)
{
    int i, j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            if (fabs(coef[i][j]) < sigma)
                coef[i][j] = 0.0;
}

//----------------------------------------------------------
//  DCT ZigZag thresholding
//----------------------------------------------------------
void ZigZagThreshold(float sigma, float **coef, int N)
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
                coef[i][j] = 0.0;
}

//---------------------------------------------------------
//---------------------------------------------------------
// PROGRAMME PRINCIPAL   ----------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
int main(int argc, char **argv)
{
    int length, width;
    char BufSystVisuImg[NBCHAR];

    //>Lecture Image
    float **Img = LoadImagePgm(NAME_IMG_IN, &length, &width);

    //>CPU Memory Allocation
    float **ImgDegraded = fmatrix_allocate_2d(length, width);
    float **ImgDenoised = fmatrix_allocate_2d(length, width);

    copy_matrix(ImgDegraded, Img, length, width);

    //>GPU Memory Allocation
    float *ImgDegraded1d = fmatrix_allocate_1d(length * width);
    float *cuImgDegraded = fmatrix_allocate_2d_device(length, width);

    copy_matrix_2d_to_1d(ImgDegraded, ImgDegraded1d, length, width);

    cudaMemcpy(cuImgDegraded, ImgDegraded1d, width * length * sizeof(float), cudaMemcpyHostToDevice); // copy of the yet to be degraded image on the GPU
    // cudaMemcpy(cuImgDenoised, ImgDenoised[0], width * length * sizeof(float), cudaMemcpyHostToDevice); // copy of the yet to be denoised image (empty at this point) on the GPU

    // textures declaration. We will be using textures for ease of use, readability and performance
    // texture<float, 2> texImgDegraded;
    // texture<float, 2> texImgDenoised;

    // binding the arrays to 2D textures
    // cudaBindTexture2D(NULL, texImgDegraded, cuImgDegraded, cudaCreateChannelDesc<float>(), width, length, width * sizeof(float));
    // cudaBindTexture2D(NULL, texImgDenoised, cuImgDenoised, cudaCreateChannelDesc<float>(), width, length, width * sizeof(float));

    //>GPU Degradation
    add_gaussian_noise_to_matrix(cuImgDegraded, length, width, SIGMA_NOISE * SIGMA_NOISE);

    // Allocate memory on CPU to store degraded image
    float *tmp = (float *)malloc(width * length * sizeof(float));

    // Bind the texture to a CUDA array
    // cudaArray *cuArrayImgDegraded;
    // cudaChannelFormatDesc channelDescImgDegraded = cudaCreateChannelDesc<float>();
    // cudaMallocArray(&cuArrayImgDegraded, &channelDescImgDegraded, width, length);
    // cudaBindTextureToArray(texImgDegraded, cuArrayImgDegraded, channelDescImgDegraded);

    // Copy the CUDA array data to CPU buffer
    cudaMemcpy(tmp, cuImgDegraded, width * length * sizeof(float), cudaMemcpyDeviceToHost);

    copy_matrix_1d_to_2d(tmp, ImgDegraded, length, width);

    // Unbind the texture
    // cudaUnbindTexture(texImgDegraded);

    printf("\n\n  Info Noise");
    printf("\n  ------------");
    printf("\n    > MSE = [%.2f]", computeMMSE(ImgDegraded, Img, length));

    //=========================================================
    //== PROG =================================================
    //=========================================================

    DctDenoise(ImgDegraded, ImgDenoised, Img, length, width, THRESHOLD);

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

    free_fmatrix_1d(tmp);
    free_fmatrix_1d(ImgDegraded1d);

    cudaFree(cuImgDegraded);
    // cudaFree(cuImgDenoised);

    // cudaUnbindTexture(texImgDegraded);
    // cudaUnbindTexture(texImgDenoised);

    // Return
    printf("\n C'est fini... \n");
    return 0;
}
