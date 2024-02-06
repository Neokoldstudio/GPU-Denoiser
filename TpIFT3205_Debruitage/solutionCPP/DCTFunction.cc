//------------------------------------------------------
// module  : DCTFunction.cc
// author  : Mignotte Max
// date    :
// version : 1.0
// language: C++
// labo    : DIRO
// note    :
//------------------------------------------------------
// quelques fonctions de reservations memoires utiles 

//------------------------------------------------
// INCLUDED FUNCTION -----------------------------
//------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>

#include "DCTFunction.h"
#include "DCT-Denoise.h"

//------------------------------------------------
// FUNCTIONS -------------------------------------
//------------------------------------------------
//---------------------------------------------------------
//  Allocation matrx 1d float
//----------------------------------------------------------
float* fmatrix_allocate_1d(int hsize)
 {
  float* matrix;
  matrix=new float[hsize]; return matrix; }

//----------------------------------------------------------
//  Allocation matrix 2d float
//----------------------------------------------------------
float** fmatrix_allocate_2d(int vsize,int hsize)
 {
  float** matrix;
  float *imptr;

  matrix=new float*[vsize];
  imptr=new  float[(hsize)*(vsize)];
  for(int i=0;i<vsize;i++,imptr+=hsize) matrix[i]=imptr;
  return matrix;
 }

//----------------------------------------------------------
// Allocation matrix 3d float
//----------------------------------------------------------
float*** fmatrix_allocate_3d(int dsize,int vsize,int hsize)
 {
  float*** matrix;

  matrix=new float**[dsize];

  for(int i=0;i<dsize;i++)
    matrix[i]=fmatrix_allocate_2d(vsize,hsize);
  return matrix;
 }

//----------------------------------------------------------
// Free Memory 1d de float
//----------------------------------------------------------
void free_fmatrix_1d(float* pmat)
{ delete[] pmat; }

//----------------------------------------------------------
// Free Memory 2d de float
//----------------------------------------------------------
void free_fmatrix_2d(float** pmat)
{ delete[] (pmat[0]);
  delete[] pmat;}

//----------------------------------------------------------
// Free Memory 3d de float
//----------------------------------------------------------
void free_fmatrix_3d(float*** pmat,int dsize)
{
 for(int i=0;i<dsize;i++)
  {
   delete[] (pmat[i][0]);
   delete[] (pmat[i]);
   }
 delete[] (pmat);
}

//----------------------------------------------------------
// copy matrix
//----------------------------------------------------------
void copy_matrix(float** mat1,float** mat2,int lgth,int wdth)
{
 int i,j;

 for(i=0;i<lgth;i++) for(j=0;j<wdth;j++)
   mat1[i][j]=mat2[i][j]; 
}

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


//---------------------------------------------------------------
// compute Moy
//---------------------------------------------------------------
void computeMoy(float** mt,int sz)
{
  int i,j;
  float moy;

  //Calcul NRJ vraissemblance
  moy=0.0;
  for(i=0;i<sz;i++) for(j=0;j<sz;j++) moy+=mt[i][j];
  moy/=(sz*sz);
 
  //Affiche
  printf("  [Moy>[%.1f]",moy);
}

//----------------------------------------------------------
// Get Length and Width
//----------------------------------------------------------
void GetLengthWidth(char* path,int* length,int* width)
{
 unsigned char var;
 int   temp;
 char* tempc;

  char stringTmp1[100];
  char stringTmp2[100];
 
  int ta1,ta2;
  FILE *fic;

  //ouverture du fichier
  fic=fopen(path,"r");
  if (fic==NULL)
    { printf("\n- Grave erreur a l'ouverture de %s -\n",path);
      exit(-1); }

  //recuperation de l'entete
  tempc=fgets(stringTmp1,100,fic);
  for(;;) { temp=fread(&var,1,1,fic); if (var==35) tempc=fgets(stringTmp2,100,fic);
            else break; }
  fseek(fic,-1,SEEK_CUR);
  temp=fscanf(fic,"%d %d",&ta1,&ta2);
   
  //enregistrement
  (*length)=ta2; 
  (*width)=ta1;

  //fermeture du fichier
  if ((temp)||(tempc)) { ; }
  fclose(fic);
}

//----------------------------------------------------------
// load pgm image 
//----------------------------------------------------------
void load_image(float** data,char* path,int length,int width)
 {
  int i,j;
  int   temp;
  char* tempc;
  unsigned char var;
  
  char header[100];
  char* ptr;
  
  int ta1,ta2,ta3;
  FILE *fic;

  //Open file
  fic=fopen(path,"r");
  if (fic==NULL)
    { printf("\n -> Grave erreur a l'ouverture de %s !\n",path);
      exit(-1); }

  tempc=fgets(header,100,fic);
  if ( (header[0]!=80) ||    /* 'P' */
       (header[1]!=53) ) {   /* '5' */
       fprintf(stderr,"Image %s is not PGM.\n",path);
       exit(1); }

  tempc=fgets(header,100,fic);
  while(header[0]=='#') tempc=fgets(header,100,fic);
 
  ta1=strtol(header,&ptr,0);
  ta2=atoi(ptr);
  tempc=fgets(header,100,fic);
  ta3=strtol(header,&ptr,0);
    
  //Load
  for(i=0;i<length;i++) for(j=0;j<width;j++)  
      { temp=fread(&var,1,1,fic);
        data[i][j]=var; }

  //Close file
  if ((temp)||(tempc)||(ta1)||(ta2)||(ta3)) { ; }
  fclose(fic);
 }


//----------------------------------------------------------
// save picture PGM 
//----------------------------------------------------------
void save_picture_pgm(char* Path_out,char* Name,float** mat,int lgth,int wdth)
 {
  int i,j;
  char buff[200];
  FILE* fuser;
  time_t tm;

  //extension
  strcpy(buff,Path_out);
  strcat(buff,Name);
  strcat(buff,".pgm");

  //open file
  fuser=fopen(buff,"w");
    if (fuser==NULL) 
        { printf(" probleme dans la sauvegarde de %s",buff); 
          exit(-1); }

  //Print
  printf("\n sauvegarde dans -> %s au format %s",buff,".pgm");

  //Save Comment
  fprintf(fuser,"P5");
  fprintf(fuser,"\n# IMG Module, %s",ctime(&tm));
  fprintf(fuser,"%d %d",wdth,lgth);
  fprintf(fuser,"\n255\n");

  //Load
     for(i=0;i<lgth;i++)
       for(j=0;j<wdth;j++) 
        fprintf(fuser,"%c",(char)mat[i][j]);
   
  //Close file
   fclose(fuser); 
 }

//----------------------------------------------------------
// Fast FilteringDCT 8x8  <simple & optimise>       
//----------------------------------------------------------
void FilteringDCT_8x8(float** imgin,float sigma,int length,int width,float** SquWin,float***  mat3d)
{
 int i,j,k,l;
 int x,y;
 int pos,posr,posc;
 float temp;
 float nb;

 //Boucle
 //------
 for(i=0;i<length;i++) for(j=0;j<width;j++) 
   {
     for(k=0;k<8;k++) for(l=0;l<8;l++) 
        { 
	 posr=i-4+k;
         posc=j-4+l;

         if (posr<0)           posr+=length;
         if (posr>(length-1))  posr-=length; 
         if (posc<0)           posc+=width;
         if (posc>(width-1))   posc-=width;

         SquWin[k][l]=imgin[posr][posc];
        }

     ddct8x8s(-1,SquWin);
     HardThreshold(sigma,SquWin,8);
     ddct8x8s(1,SquWin); 
    
     x=(i%8);
     y=(j%8);
     pos=((x*8)+y);

     for(k=0;k<8;k++) for(l=0;l<8;l++) 
        { 
          posr=i-4+k;
          posc=j-4+l;

          if (posr<0)           posr+=length;
          if (posr>(length-1))  posr-=length; 
          if (posc<0)           posc+=width;
          if (posc>(width-1))   posc-=width;

          mat3d[pos][posr][posc]=SquWin[k][l]; 
         }
   }

 //Moyennage
 //---------
 for(i=0;i<length;i++) for(j=0;j<width;j++)
    {
     temp=0.0; nb=0.0;
     for(k=0;k<64;k++)
       if (mat3d[k][i][j]>0) { nb++; temp+=mat3d[k][i][j]; }
        
     if (nb) { temp/=nb;  imgin[i][j]=temp; }   
     
    }
}

//----------------------------------------------------------
//----------------------------------------------------------
// Fast FilteringDCT 8x8  <simple> <ovl>
//----------------------------------------------------------
//----------------------------------------------------------
void FilteringDCT_8x8_(float** imgin,float sigma,int length,int width,float** SquWin,float***  mat3d)
{
 int i,j;
 int k,l;
 int x,y;
 int pos;
 float temp;
 float nb;
 int posr,posc;
 int overlap;

 //Initialisation
 //--------------
 //>Record
 overlap=OVERLAP;       

 //>Init
 for(k=0;k<64;k++)
 for(i=0;i<length;i++) for(j=0;j<width;j++) mat3d[k][i][j]=-1.0;
   
 //Loop
 //----
 for(i=0;i<length;i+=overlap) for(j=0;j<width;j+=overlap) 
   {
     for(k=0;k<8;k++) for(l=0;l<8;l++) 
        { 
	 posr=i-4+k;
         posc=j-4+l;

         if (posr<0)           posr+=length;
         if (posr>(length-1))  posr-=length; 
 
         if (posc<0)           posc+=width;
         if (posc>(width-1))   posc-=width;

         SquWin[k][l]=imgin[posr][posc];
        }

     ddct8x8s(-1,SquWin);
     if (HARD_THRESHOLD)  HardThreshold(sigma,SquWin,8);
     if (!HARD_THRESHOLD) ZigZagThreshold(sigma,SquWin,8);
     ddct8x8s(1,SquWin); 
    
     x=(i%8);
     y=(j%8);
     pos=((x*8)+y);

     for(k=0;k<8;k++) for(l=0;l<8;l++) 
        { 
          posr=i-4+k;
          posc=j-4+l;

          if (posr<0)           posr+=length;
          if (posr>(length-1))  posr-=length; 
 
          if (posc<0)           posc+=width;
          if (posc>(width-1))   posc-=width;

          if  (mat3d[pos][posr][posc]!=-1) printf("!"); 

          mat3d[pos][posr][posc]=SquWin[k][l]; 
         }
   }

 //Averaging
 //---------
 for(i=0;i<length;i++) for(j=0;j<width;j++)
    {
     temp=0.0; nb=0.0;
     for(k=0;k<64;k++)
       if (mat3d[k][i][j]>0.0)  { nb++; temp+=mat3d[k][i][j]; }
        
     if (nb) { temp/=nb;  imgin[i][j]=temp; }
    } 
}

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

//----------------------------------------------------------
//  DCT thresholding         
//----------------------------------------------------------
void HardThreshold(float sigma,float** coef,int N)
{
 int i,j; 

 for(i=0;i<N;i++) for(j=0;j<N;j++) if (fabs(coef[i][j])<sigma) coef[i][j]=0.0;
}

//----------------------------------------------------------
//  DCT ZigZag thresholding         
//----------------------------------------------------------
void ZigZagThreshold(float sigma,float** coef,int N)
{
 int result[8][8];
 int i=0;
 int j=0;
 int d=-1; 
 int start=0;
 int end=63;

    //ZigZag Matrix
    do
    {
     result[i][j]=start++;
     result[N-i-1][N-j-1]=end--;
 
     i+=d; j-=d;
     if (i<0)
        {
          i++; d=-d; 
        }
        else if (j<0)
        {
          j++; d = -d; 
        }
    } while (start<end);
    if (start==end)
        result[i][j]=start;

    //Seuillage
    for(i=0;i<N;i++) for(j=0;j<N;j++)
    if (result[i][j]>=sigma) coef[i][j]=0.0;
}

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
 if (noise>255) noise=255;
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

