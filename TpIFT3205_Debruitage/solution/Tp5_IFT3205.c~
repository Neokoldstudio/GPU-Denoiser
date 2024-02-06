//------------------------------------------------------
// Prog    : Tp5_IFT3205                          
// Auteur  :                                            
// Date    :                                  
// version :                                             
// langage : C                                          
// labo    : DIRO                                       
//------------------------------------------------------


//------------------------------------------------
// FICHIERS INCLUS -------------------------------
//------------------------------------------------
#include "FonctionDemo5.h"

//------------------------------------------------
// DEFINITIONS -----------------------------------
//------------------------------------------------
#define NAME_VISUALISER "display "
#define NAME_IMG_IN  "lena512"
#define NAME_IMG_OUT  "lena512_Restored"
#define NAME_IMG_DEG  "lena512_Degraded"

//------------------------------------------------
// PROTOTYPE DE FONCTIONS  -----------------------
//------------------------------------------------

//>Main Function
void DctDenoise(float**,float**,float**,int,int,float);

//>Gestion
void copy_matrix(float**,float**,int,int);
void FilteringDCT_8x8_(float**,float,int,int,float**,float***);
void FilteringDCT_8x8(float**,float,int,int,float**,float***);
void HardThreshold(float,float**,int);
void ZigZagThreshold(float,float**,int);

//------------------------------------------------
// CONSTANTS -------------------------------------
//------------------------------------------------
//-------------
// Question 2.2
// ------------
// #define SIGMA_NOISE   30 
// #define NB_ITERATIONS  1 
// #define THRESHOLD      6 
// #define OVERLAP        8 
// #define HARD_THRESHOLD  0 
//
// MSE=157.00
// 

//-------------
// Question 2.3
// ------------
// #define SIGMA_NOISE    30 
// #define NB_ITERATIONS   1 
// #define THRESHOLD     100 
// #define OVERLAP         8  
// #define HARD_THRESHOLD  1 
//
// MSE=140.28
// 

//-------------
// Question 2.4
// ------------
#define SIGMA_NOISE    30
#define NB_ITERATIONS   1
#define THRESHOLD      90
#define OVERLAP         1
#define HARD_THRESHOLD  1
//
// MSE=65.04
//

//-------------
// Question X.X
// ------------
/* #define SIGMA_NOISE     30 */
/* #define  NB_ITERATIONS   5 */
/* #define THRESHOLD       30 */
/* #define OVERLAP          1  */
/* #define HARD_THRESHOLD   1 */
//
// MSE=58.06
//


#define  ZOOM 1
#define  QUIT 0    


//------------------------------------------------
//------------------------------------------------
// FONCTIONS  ------------------------------------   
//------------------------------------------------
//------------------------------------------------
//----------------------------------------------------------
// IterDctDenoise      
//----------------------------------------------------------     
void DctDenoise(float**  DataDegraded,float** DataFiltered,float** Data,int lgth,int wdth,float Thresh)
{
 int k;
 int SizeWindow;
 char Name_img[NBCHAR];

 //Parameter
 SizeWindow=8;

 //Info
 //----;
 printf("\n   ---------------- ");
 printf("\n    IterDctDenoise ");
 printf("\n   ----------------");
 printf("\n  Length:Width [%d][%d]",lgth,wdth);
 printf("\n  -----------------------");
 printf("\n   >> SigmaNoise = [%d]",SIGMA_NOISE);
 printf("\n  -----------------------");
 printf("\n  Threshold_Dct  > [%.1d]",THRESHOLD);
 printf("\n  Size Window    > [%d]",SizeWindow);
 printf("\n  Overlap        > [%d]",OVERLAP); 
 printf("\n\n");

 //Allocation Memoire
 float** SquWin=fmatrix_allocate_2d(SizeWindow,SizeWindow);
 float*** mat3d=fmatrix_allocate_3d(SizeWindow*SizeWindow,lgth,wdth);

 //Init
 copy_matrix(DataFiltered,DataDegraded,lgth,wdth);

 //>Loop-DEnoising
 for(k=0;k<NB_ITERATIONS;k++)
   { FilteringDCT_8x8_(DataFiltered,THRESHOLD,lgth,wdth,SquWin,mat3d); 
     printf("\n   > MSE >> [%.5f]",computeMMSE(DataFiltered,Data,lgth)); }

 //Free memory
 if (SquWin) free_fmatrix_2d(SquWin); 
 if (mat3d)  free_fmatrix_3d(mat3d,SizeWindow*SizeWindow);
}

//---------------//
//--- GESTION ---//
//---------------//
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
 int end=(N*N)-1;

    //>ZigZag Matrix
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

    //>Seuillage
    for(i=0;i<N;i++) for(j=0;j<N;j++)
    if (result[i][j]>=sigma) coef[i][j]=0.0;
}


//---------------------------------------------------------
//---------------------------------------------------------
// PROGRAMME PRINCIPAL   ----------------------------------                     
//---------------------------------------------------------
//---------------------------------------------------------
int main(int argc,char** argv)
{
 int length,width;
 char BufSystVisuImg[NBCHAR];

 //>Lecture Image 
 float** Img=LoadImagePgm(NAME_IMG_IN,&length,&width);
 
 //>Allocation memory
 float** ImgDegraded=fmatrix_allocate_2d(length,width);
 float** ImgDenoised=fmatrix_allocate_2d(length,width);  

 //>Degradation 
 copy_matrix(ImgDegraded,Img,length,width);
 add_gaussian_noise(ImgDegraded,length,width,SIGMA_NOISE*SIGMA_NOISE);
 
 printf("\n\n  Info Noise");
 printf("\n  ------------");
 printf("\n    > MSE = [%.2f]",computeMMSE(ImgDegraded,Img,length)); 
 

 //=========================================================
 //== PROG =================================================
 //=========================================================

 DctDenoise(ImgDegraded,ImgDenoised,Img,length,width,THRESHOLD);

  //---------------------------------------------
  // SAUVEGARDE 
  // -------------------
  // L'image dégradée             > ImgDegraded
  // Le resultat du debruitage    > ImgFiltered
  //----------------------------------------------
  SaveImagePgm(NAME_IMG_DEG,ImgDegraded,length,width);
  SaveImagePgm(NAME_IMG_OUT,ImgDenoised,length,width);  

  //>Visu Img
  strcpy(BufSystVisuImg,NAME_VISUALISER);
  strcat(BufSystVisuImg,NAME_IMG_IN);
  strcat(BufSystVisuImg,".pgm&");
  printf("\n > %s",BufSystVisuImg);
  system(BufSystVisuImg);

  //Visu ImgDegraded
  strcpy(BufSystVisuImg,NAME_VISUALISER);
  strcat(BufSystVisuImg,NAME_IMG_DEG);
  strcat(BufSystVisuImg,".pgm&");
  printf("\n > %s",BufSystVisuImg);
  system(BufSystVisuImg);
  
  //Visu ImgFiltered
  strcpy(BufSystVisuImg,NAME_VISUALISER);
  strcat(BufSystVisuImg,NAME_IMG_OUT);
  strcat(BufSystVisuImg,".pgm&");
  printf("\n > %s",BufSystVisuImg);
  system(BufSystVisuImg);

    
//--------------- End -------------------------------------     
//----------------------------------------------------------

  //Liberation memoire pour les matrices
  if (Img)          free_fmatrix_2d(Img);
  if (ImgDegraded)  free_fmatrix_2d(ImgDegraded);
  if (ImgDenoised)  free_fmatrix_2d(ImgDenoised);
  
  //Return
  printf("\n C'est fini... \n");; 
  return 0;
 }
 


