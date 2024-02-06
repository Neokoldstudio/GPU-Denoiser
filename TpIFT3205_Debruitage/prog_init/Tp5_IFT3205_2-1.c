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
#define NAME_VISUALISER "open "
#define NAME_IMG_IN  "lena512"
#define NAME_IMG_OUT  "lena512_Restored"
#define NAME_IMG_DEG  "lena512_Degraded"

//------------------------------------------------
// PROTOTYPE DE FONCTIONS  -----------------------
//------------------------------------------------
//----------------------------------------------------------
// copy matrix
//----------------------------------------------------------
void copy_matrix(float** mat1,float** mat2,int lgth,int wdth)
{
 int i,j;

 for(i=0;i<lgth;i++) for(j=0;j<wdth;j++) mat1[i][j]=mat2[i][j];
}



//------------------------------------------------
// CONSTANTES ------------------------------------
//------------------------------------------------
#define SIGMA_NOISE  30 

//------------------------------------------------
//------------------------------------------------
// FONCTIONS  ------------------------------------   
//------------------------------------------------
//------------------------------------------------


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

 // .....

  //---------------------------------------------
  // SAUVEGARDE 
  // -------------------
  // L'image d�grad�e             > ImgDegraded
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
 


//----------------------------------------------------------
// Allocation matrix 3d float
// --------------------------
//
// float*** fmatrix_allocate_3d(int dsize,int vsize,int hsize)
// > Alloue dynamiquement un tableau 3D [dsize][vsize][hsize]
//
//-----------------------------------------------------------

//----------------------------------------------------------
//  ddct8x8s(int isgn, float** tab)
// ---------------------------------
//
// Fait la TCD (sgn=-1) sur un tableau 2D tab[8][8]
// Fait la TCD inverse (sgn=1) sur un tableau 2D tab[8][8]
//
//-----------------------------------------------------------
