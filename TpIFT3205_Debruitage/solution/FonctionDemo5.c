//------------------------------------------------------
// module  : FonctionDemo5.c
// auteur  : Mignotte Max
// date    :
// version : 1.0
// langage : C++
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

#include "FonctionDemo5.h"


//--------------------------//
//-- Matrice de Flottant --//
//--------------------------//
//---------------------------------------------------------
//  Alloue de la memoire pour une matrice 1d de float      
//---------------------------------------------------------
float* fmatrix_allocate_1d(int hsize)
 {
  float* matrix;

  matrix=(float*)malloc(sizeof(float)*hsize);
  if (matrix==NULL) printf("probleme d'allocation memoire");

  return matrix;
 }

//----------------------------------------------------------
//  Alloue de la memoire pour une matrice 2d de float       
//----------------------------------------------------------
float** fmatrix_allocate_2d(int vsize,int hsize)
 {
  int i;
  float** matrix;
  float *imptr;

  matrix=(float**)malloc(sizeof(float*)*vsize);
  if (matrix==NULL) printf("probleme d'allocation memoire");

  imptr=(float*)malloc(sizeof(float)*hsize*vsize);
  if (imptr==NULL) printf("probleme d'allocation memoire");

  for(i=0;i<vsize;i++,imptr+=hsize) matrix[i]=imptr;
  return matrix;
 }

//----------------------------------------------------------
// Allocation matrix 3d float
//----------------------------------------------------------
float*** fmatrix_allocate_3d(int dsize,int vsize,int hsize)
 {
  int i;
  float*** matrix;

  matrix=(float***)malloc(sizeof(float**)*dsize); 

  for(i=0;i<dsize;i++)
    matrix[i]=fmatrix_allocate_2d(vsize,hsize);
  return matrix;
 }

//----------------------------------------------------------
// Libere la memoire de la matrice 1d de float              
//----------------------------------------------------------
void free_fmatrix_1d(float* pmat)
 {
  free(pmat);
 }

//----------------------------------------------------------
// Libere la memoire de la matrice 2d de float              
//----------------------------------------------------------
void free_fmatrix_2d(float** pmat)
 {
  free(pmat[0]);
  free(pmat);
 }

//----------------------------------------------------------
// Free Memory 3d de float
//----------------------------------------------------------
void free_fmatrix_3d(float*** pmat,int dsize)
{
 int i;
 for(i=0;i<dsize;i++)
  {
   free(pmat[i][0]);
   free(pmat[i]);
   }
 free(pmat);
}

//--------------------//
//-- LOAD/SAVE/FILE --//
//--------------------//
//----------------------------------------------------------
// Chargement de l'image de nom <name> (en pgm)             
//----------------------------------------------------------
float** LoadImagePgm(char* name,int *length,int *width)
 {
  int i,j;
  unsigned char var;
  char buff[NBCHAR];
  int tmp;
  char* ptmp;
  float** mat;

  char stringTmp1[NBCHAR],stringTmp2[NBCHAR];
 
  int ta1,ta2,ta3;
  FILE *fic;

  //-----nom du fichier pgm-----
  strcpy(buff,name);
  strcat(buff,".pgm");
  printf("---> Ouverture de %s",buff);

  //----ouverture du fichier----
  fic=fopen(buff,"r");
  if (fic==NULL)
    { printf("\n- Grave erreur a l'ouverture de %s  -\n",buff);
      exit(-1); }

  //--recuperation de l'entete--
  ptmp=fgets(stringTmp1,100,fic);
  ptmp=fgets(stringTmp2,100,fic);
  tmp=fscanf(fic,"%d %d",&ta1,&ta2);
  tmp=fscanf(fic,"%d\n",&ta3);

  //--affichage de l'entete--
  printf("\n\n--Entete--");
  printf("\n----------");
  printf("\n%s%s%d %d \n%d\n",stringTmp1,stringTmp2,ta1,ta2,ta3);

  *length=ta1;
  *width=ta2;
  mat=fmatrix_allocate_2d(*length,*width);
   
  //--chargement dans la matrice--
     for(i=0;i<*length;i++)
      for(j=0;j<*width;j++)  
        { tmp=fread(&var,1,1,fic);
          mat[i][j]=var; }

   //---fermeture du fichier---
  fclose(fic);

  return(mat);
 }


//----------------------------------------------------------
// Sauvegarde de l'image de nom <name> au format pgm        
//----------------------------------------------------------
void SaveImagePgm(char* name,float** mat,int length,int width)
 {
  int i,j;
  char buff[NBCHAR];
  FILE* fic;
  time_t tm;

  //--extension--
  strcpy(buff,name);
  strcat(buff,".pgm");

  //--ouverture fichier--
  fic=fopen(buff,"w");
    if (fic==NULL) 
        { printf(" Probleme dans la sauvegarde de %s",buff); 
          exit(-1); }
  printf("\n Sauvegarde de %s au format pgm\n",name);

  //--sauvegarde de l'entete--
  fprintf(fic,"P5");
  if ((ctime(&tm))==NULL) fprintf(fic,"\n#\n");
  else fprintf(fic,"\n# IMG Module, %s",ctime(&tm));
  fprintf(fic,"%d %d",width,length);
  fprintf(fic,"\n255\n");

  //--enregistrement--
     for(i=0;i<length;i++)
      for(j=0;j<width;j++) 
        fprintf(fic,"%c",(char)mat[i][j]);
   
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
#define C8_1R   0.49039264020161522456
#define C8_1I   0.09754516100806413392
#define C8_2R   0.46193976625564337806
#define C8_2I   0.19134171618254488586
#define C8_3R   0.41573480615127261854
#define C8_3I   0.27778511650980111237
#define C8_4R   0.35355339059327376220
#define W8_4R   0.70710678118654752440

void ddct8x8s(int isgn, float **a)
{
    int j;
    float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
    float xr, xi;
    
    if (isgn < 0) {
        for (j = 0; j <= 7; j++) {
            x0r = a[0][j] + a[7][j];
            x1r = a[0][j] - a[7][j];
            x0i = a[2][j] + a[5][j];
            x1i = a[2][j] - a[5][j];
            x2r = a[4][j] + a[3][j];
            x3r = a[4][j] - a[3][j];
            x2i = a[6][j] + a[1][j];
            x3i = a[6][j] - a[1][j];
            xr = x0r + x2r;
            xi = x0i + x2i;
            a[0][j] = C8_4R * (xr + xi);
            a[4][j] = C8_4R * (xr - xi);
            xr = x0r - x2r;
            xi = x0i - x2i;
            a[2][j] = C8_2R * xr - C8_2I * xi;
            a[6][j] = C8_2R * xi + C8_2I * xr;
            xr = W8_4R * (x1i - x3i);
            x1i = W8_4R * (x1i + x3i);
            x3i = x1i - x3r;
            x1i += x3r;
            x3r = x1r - xr;
            x1r += xr;
            a[1][j] = C8_1R * x1r - C8_1I * x1i;
            a[7][j] = C8_1R * x1i + C8_1I * x1r;
            a[3][j] = C8_3R * x3r - C8_3I * x3i;
            a[5][j] = C8_3R * x3i + C8_3I * x3r;
        }
        for (j = 0; j <= 7; j++) {
            x0r = a[j][0] + a[j][7];
            x1r = a[j][0] - a[j][7];
            x0i = a[j][2] + a[j][5];
            x1i = a[j][2] - a[j][5];
            x2r = a[j][4] + a[j][3];
            x3r = a[j][4] - a[j][3];
            x2i = a[j][6] + a[j][1];
            x3i = a[j][6] - a[j][1];
            xr = x0r + x2r;
            xi = x0i + x2i;
            a[j][0] = C8_4R * (xr + xi);
            a[j][4] = C8_4R * (xr - xi);
            xr = x0r - x2r;
            xi = x0i - x2i;
            a[j][2] = C8_2R * xr - C8_2I * xi;
            a[j][6] = C8_2R * xi + C8_2I * xr;
            xr = W8_4R * (x1i - x3i);
            x1i = W8_4R * (x1i + x3i);
            x3i = x1i - x3r;
            x1i += x3r;
            x3r = x1r - xr;
            x1r += xr;
            a[j][1] = C8_1R * x1r - C8_1I * x1i;
            a[j][7] = C8_1R * x1i + C8_1I * x1r;
            a[j][3] = C8_3R * x3r - C8_3I * x3i;
            a[j][5] = C8_3R * x3i + C8_3I * x3r;
        }
    } else {
        for (j = 0; j <= 7; j++) {
            x1r = C8_1R * a[1][j] + C8_1I * a[7][j];
            x1i = C8_1R * a[7][j] - C8_1I * a[1][j];
            x3r = C8_3R * a[3][j] + C8_3I * a[5][j];
            x3i = C8_3R * a[5][j] - C8_3I * a[3][j];
            xr = x1r - x3r;
            xi = x1i + x3i;
            x1r += x3r;
            x3i -= x1i;
            x1i = W8_4R * (xr + xi);
            x3r = W8_4R * (xr - xi);
            xr = C8_2R * a[2][j] + C8_2I * a[6][j];
            xi = C8_2R * a[6][j] - C8_2I * a[2][j];
            x0r = C8_4R * (a[0][j] + a[4][j]);
            x0i = C8_4R * (a[0][j] - a[4][j]);
            x2r = x0r - xr;
            x2i = x0i - xi;
            x0r += xr;
            x0i += xi;
            a[0][j] = x0r + x1r;
            a[7][j] = x0r - x1r;
            a[2][j] = x0i + x1i;
            a[5][j] = x0i - x1i;
            a[4][j] = x2r - x3i;
            a[3][j] = x2r + x3i;
            a[6][j] = x2i - x3r;
            a[1][j] = x2i + x3r;
        }
        for (j = 0; j <= 7; j++) {
            x1r = C8_1R * a[j][1] + C8_1I * a[j][7];
            x1i = C8_1R * a[j][7] - C8_1I * a[j][1];
            x3r = C8_3R * a[j][3] + C8_3I * a[j][5];
            x3i = C8_3R * a[j][5] - C8_3I * a[j][3];
            xr = x1r - x3r;
            xi = x1i + x3i;
            x1r += x3r;
            x3i -= x1i;
            x1i = W8_4R * (xr + xi);
            x3r = W8_4R * (xr - xi);
            xr = C8_2R * a[j][2] + C8_2I * a[j][6];
            xi = C8_2R * a[j][6] - C8_2I * a[j][2];
            x0r = C8_4R * (a[j][0] + a[j][4]);
            x0i = C8_4R * (a[j][0] - a[j][4]);
            x2r = x0r - xr;
            x2i = x0i - xi;
            x0r += xr;
            x0i += xi;
            a[j][0] = x0r + x1r;
            a[j][7] = x0r - x1r;
            a[j][2] = x0i + x1i;
            a[j][5] = x0i - x1i;
            a[j][4] = x2r - x3i;
            a[j][3] = x2r + x3i;
            a[j][6] = x2i - x3r;
            a[j][1] = x2i + x3r;
        }
    }
}

//-------------------//
//--- DEGRADATION ---//
//-------------------//
//----------------------------------------------------------
//  Gaussian noisee  
//----------------------------------------------------------
float gaussian_noise(float var,float mean)
{
 float noise,theta;

 //Noise generation 
 noise=sqrt(-2*var*log(1.0-((float)rand()/RAND_MAX)));
 theta=(float)rand()*1.9175345E-4-PI;
 noise=noise*cos(theta);
 noise+=mean;
 if (noise>GREY_LEVEL) noise=GREY_LEVEL;
 if (noise<0) noise=0;
 return noise;
}

//----------------------------------------------------------
//  Add Gaussian noise 
//----------------------------------------------------------
void add_gaussian_noise(float** mat,int lgth,int wdth,float var)
{
 int i,j;

 //Loop
 for(i=0;i<lgth;i++) for(j=0;j<wdth;j++)
 if (var!=0.0) mat[i][j]=gaussian_noise(var,mat[i][j]);
}

//--------------//
//--- MESURE ---//
//--------------//
//----------------------------------------------------------
// compute MMSE              
//----------------------------------------------------------     
float computeMMSE(float** mat1,float** mat2,int sz)
{
 int i,j;
 float mmse;

 //Boucle 
 mmse=0.0;
 for(i=0;i<sz;i++) for(j=0;j<sz;j++) mmse+=CARRE(mat2[i][j]-mat1[i][j]);

 mmse/=(CARRE(sz));

 //retour
 return mmse;
}

