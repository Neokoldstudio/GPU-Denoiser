//------------------------------------------------------
// module  : f_DCT-Denoise.cc
// author  : Mignotte Max
// date    : 
// version : 1.0
// language: C++
// labo    : DIRO
// note    : 
//------------------------------------------------------
//

//------------------------------------------------
// INCLUDED FILES --------------------------------
//------------------------------------------------
#include "DCT-Denoise.h"

//------------------------------------------------
// FUNCTIONS CLASS DCTDenoise --------------------
//------------------------------------------------
//----------------------------------------------------------
// Constructor
//----------------------------------------------------------
DCTDenoise::DCTDenoise(float** mat,int lgth,int wdth)
 {
   int i,j;

   //record Info
   length=lgth; width=wdth;

   //Allocation memory
   DataDegraded=fmatrix_allocate_2d(length,width);
   DataFiltered=fmatrix_allocate_2d(length,width);  
   DataWithoutNoise=fmatrix_allocate_2d(length,width); 

   //Load
   for(i=0;i<length;i++) for(j=0;j<width;j++)
     DataDegraded[i][j]=mat[i][j];
 }

//----------------------------------------------------------
// Destructor
//----------------------------------------------------------
DCTDenoise::~DCTDenoise()
 { 
  //Free memory
  if (DataDegraded)      free_fmatrix_2d(DataDegraded); 
  if (DataFiltered)      free_fmatrix_2d(DataFiltered);
  if (DataWithoutNoise)  free_fmatrix_2d(DataWithoutNoise);
 }

//-------------------//
// Options ----------//
//-------------------//
//----------------------------------------------------------
// Options        
//----------------------------------------------------------
void DCTDenoise::Options(float** mat,float Sg,float ThreshDct)
{
 int i,j;

 //Record Info
  SigmaNoise=Sg;
  Threshold_Dct=ThreshDct;
  printf("\n  SigmaNoise    > [%.2f]",SigmaNoise);
  printf("\n  Threshold_Dct > [%.2f]",ThreshDct);
  
 //Record Img 
 for(i=0;i<length;i++) for(j=0;j<width;j++)
   DataWithoutNoise[i][j]=mat[i][j];
}


//-------------------//
// Methods ----------//
//-------------------//
//----------------------------------------------------------
// DCT Denoising        
//----------------------------------------------------------     
void DCTDenoise::IterDctDenoise( )
{
 int k;

 int SizeWindow;

 //Parameter
 SizeWindow=8;

 //Info
 //----;
 printf("\n   ------------------ ");
 printf("\n    IterDctDenoise ");
 printf("\n   ------------------ \n");
 printf("\n  Length:[%d]   Width:[%d]",length,width);
 printf("\n  -----------------------");
 printf("\n   >> Sigma Noise  = [%.2f]",SigmaNoise);
 printf("\n   >> Thresh _Dct  = [%.2f]",Threshold_Dct);
 printf("\n  -----------------------");
 printf("\n  Size Window    > [%d]",SizeWindow);
 printf("\n  Overlap        > [%d]",OVERLAP); 
 printf("\n\n");

  //Allocation Memoire
 float** SquWin=fmatrix_allocate_2d(SizeWindow,SizeWindow);
 float*** mat3d=fmatrix_allocate_3d(SizeWindow*SizeWindow,length,width);

 //Init
 copy_matrix(DataFiltered,DataDegraded,length,width);

 //Loop -------------
 for(k=0;k<NB_ITERATIONS;k++)
    { 
      FilteringDCT_8x8_(DataFiltered,Threshold_Dct,length,width,SquWin,mat3d); 
      printf("\n   > MSE >> [%.5f]",computeMMSE(DataFiltered,DataWithoutNoise,length)); 
    }
 //--------------------

 save_picture_pgm((char*)"",(char*)"RESTORED",DataFiltered,length,width);

  //Free memory
  if (SquWin) free_fmatrix_2d(SquWin); 
  if (mat3d)  free_fmatrix_3d(mat3d,SizeWindow*SizeWindow);

}

