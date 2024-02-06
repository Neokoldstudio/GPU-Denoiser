//------------------------------------------------------
// module  : DCT-Denoise.cc
// author  : Mignotte Max
// date    : 
// version : 1.0
// language: C++
// note    :
//------------------------------------------------------
//  

//------------------------------------------------
// INCLDED FILES ---------------------------------
//------------------------------------------------

#include "DCT-Denoise.h"

//------------------------------------------------
// CONSTANTS -------------------------------------
//------------------------------------------------
const int ZOOM=1;
const int QUIT=0;
const float SIGMA_NOISE=30;
const float THRESHOLD_DCT=3*SIGMA_NOISE;

//------------------------------------------------
// GLOBAL VARIABLES ------------------------------
//------------------------------------------------
char Name_img[100];

int flag_save;
int flag_quit;
int zoom;

int length,width;
float SigmaNoise;
float ThresholdDCT;

//-------------------------
//--- Windows -------------
//-------------------------
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/cursorfont.h>

Display   *display;
int	  screen_num;
int 	  depth;
Window	  root;
Visual*	  visual;
GC	  gc;

/************************************************************************/
/* OPEN_DISPLAY()							*/
/************************************************************************/
int open_display()
{
  if ((display=XOpenDisplay(NULL))==NULL)
   { printf("Connection impossible\n");
     return(-1); }

  else
   { screen_num=DefaultScreen(display);
     visual=DefaultVisual(display,screen_num);
     depth=DefaultDepth(display,screen_num);
     root=RootWindow(display,screen_num);
     return 0; }
}

/************************************************************************/
/* FABRIQUE_WINDOW()							*/
/* Cette fonction crée une fenetre X et l'affiche à l'écran.	        */
/************************************************************************/
Window fabrique_window(char *nom_fen,int x,int y,int width,int height,int zoom)
{
  Window                 win;
  XSizeHints      size_hints;
  XWMHints          wm_hints;
  XClassHint     class_hints;
  XTextProperty  windowName, iconName;

  char *name=nom_fen;

  if(zoom<0) { width/=-zoom; height/=-zoom; }
  if(zoom>0) { width*=zoom;  height*=zoom;  }

  win=XCreateSimpleWindow(display,root,x,y,width,height,1,0,255);

  size_hints.flags=PPosition|PSize|PMinSize;
  size_hints.min_width=width;
  size_hints.min_height=height;

  XStringListToTextProperty(&name,1,&windowName);
  XStringListToTextProperty(&name,1,&iconName);
  wm_hints.initial_state=NormalState;
  wm_hints.input=True;
  wm_hints.flags=StateHint|InputHint;
  class_hints.res_name=nom_fen;
  class_hints.res_class=nom_fen;

  XSetWMProperties(display,win,&windowName,&iconName,
                   NULL,0,&size_hints,&wm_hints,&class_hints);

  gc=XCreateGC(display,win,0,NULL);

  XSelectInput(display,win,ExposureMask|KeyPressMask|ButtonPressMask| 
               ButtonReleaseMask|ButtonMotionMask|PointerMotionHintMask| 
               StructureNotifyMask);

  XMapWindow(display,win);
  return(win);
}

/****************************************************************************/
/* CREE_XIMAGE2()							    */
/* Crée une XImage à partir d'un tableau de float                           */
/* L'image peut subir un zoom.						    */
/****************************************************************************/
XImage* cree_Ximage2(float** mat,int z,int length,int width)
{
  int lgth,wdth,lig,col,zoom_col,zoom_lig;
  float somme;
  unsigned char	 pix;
  unsigned char* dat;
  XImage* imageX;

  /*Zoom positiv*/
  /*------------*/
  if (z>0)
  {
   lgth=length*z;
   wdth=width*z;

   dat=(unsigned char*)malloc(lgth*(wdth*4)*sizeof(unsigned char));
   if (dat==NULL)
      { printf("Impossible d'allouer de la memoire.");
        exit(-1); }

  for(lig=0;lig<lgth;lig=lig+z) for(col=0;col<wdth;col=col+z)
   { 
    pix=(unsigned char)mat[lig/z][col/z];
    for(zoom_lig=0;zoom_lig<z;zoom_lig++) for(zoom_col=0;zoom_col<z;zoom_col++)
      { 
       dat[((lig+zoom_lig)*wdth*4)+((4*(col+zoom_col))+0)]=pix;
       dat[((lig+zoom_lig)*wdth*4)+((4*(col+zoom_col))+1)]=pix;
       dat[((lig+zoom_lig)*wdth*4)+((4*(col+zoom_col))+2)]=pix;
       dat[((lig+zoom_lig)*wdth*4)+((4*(col+zoom_col))+3)]=pix; 
       }
    }
  } /*--------------------------------------------------------*/

  /*Zoom negatifv*/
  /*------------*/
  else
  {
   z=-z;
   lgth=(length/z);
   wdth=(width/z);

   dat=(unsigned char*)malloc(lgth*(wdth*4)*sizeof(unsigned char));
   if (dat==NULL)
      { printf("Impossible d'allouer de la memoire.");
        exit(-1); }

  for(lig=0;lig<(lgth*z);lig=lig+z) for(col=0;col<(wdth*z);col=col+z)
   {  
    somme=0.0;
    for(zoom_lig=0;zoom_lig<z;zoom_lig++)
       { for(zoom_col=0;zoom_col<z;zoom_col++)
	 somme+=mat[lig+zoom_lig][col+zoom_col]; }
           
     somme/=(z*z);    
     dat[((lig/z)*wdth*4)+((4*(col/z))+0)]=(unsigned char)somme;
     dat[((lig/z)*wdth*4)+((4*(col/z))+1)]=(unsigned char)somme;
     dat[((lig/z)*wdth*4)+((4*(col/z))+2)]=(unsigned char)somme;
     dat[((lig/z)*wdth*4)+((4*(col/z))+3)]=(unsigned char)somme; 
   }
  } /*--------------------------------------------------------*/

  imageX=XCreateImage(display,visual,depth,ZPixmap,0,(char*)dat,wdth,lgth,16,wdth*4);
  return (imageX);
}

//-------------------------
//--- Functions -----------
//-------------------------
//---------------------------------------------------------
// Read arguments      
//---------------------------------------------------------
void read_arguments(int argc,char** argv)
{
   int i;

   //Options
  if (argc<2) 
    { printf("\n Usage : %s [Img]",argv[0]); 
      printf("\n Options are : default value indicated in ()");
      printf("\n");
      printf("\n Degradation-");
      printf("\n -------------");
      printf("\n -n [SigmaNoise] (Gaussian noise standart deviation)  > %.3f",SIGMA_NOISE); 
      printf("\n -t [Threshold DCT] > %.3f",3*SIGMA_NOISE);
      printf("\n");
      printf("\n -z Zoom  (%d)",ZOOM);
      printf("\n -q Quit  (Without)"); 
      printf("\n -s Save (Ok)"); 
      printf("\n ----");
      printf("\n Examples:");
      printf("\n %s lena512.pgm -n 30 -t 90",argv[0]); 
      printf("\n\n\n\n");
      printf("\n\n");
      exit(-1); }

      //Load File Image
      strcpy(Name_img,argv[1]);      

   //Loop
   for(i=2;i<argc;i++)
     {   
          switch(argv[i][1])
          {
           case 'n': SigmaNoise=atof(argv[++i]);    break;
	   case 't': ThresholdDCT=atof(argv[++i]);  break;
           case 'z': zoom=atoi(argv[++i]);          break;
           case 's': flag_save=1;                   break; 
           case 'q': flag_quit=1;                   break;
           case '?':                                break;                
          }    
     } 
 }


//----------------------------------------------------------
//----------------------------------------------------------
// MAIN PROGRAM --------------------------------------------
//----------------------------------------------------------
//----------------------------------------------------------
int main(int argc,char** argv)
{
 int   flag;

 //For Xwindow
 //------------
 XEvent ev;
 Window win_ppicture;
 Window win_ppicture_noise;
 Window win_ppicture_result;
 XImage *x_ppicture;
 XImage *x_ppicture_noise;
 XImage *x_ppicture_result;
 char nomfen_ppicture[1000];
 char nomfen_ppicture_noise[1000]; 
 char nomfen_ppicture_result[1000];

 //Initialization
 //--------------
 SigmaNoise=SIGMA_NOISE;
 ThresholdDCT=THRESHOLD_DCT;
 zoom=ZOOM;
 flag=0;

 //Read Arguments
 //--------------
 read_arguments(argc,argv);

 //Print Options
 //---------------
 printf("\n\n\n");
 printf("\n\n Info: ");
 printf("\n -----");
 printf("\n Load File : <%s>",Name_img);
 printf("\n -----");
 printf("\n Sigma (standart deviation) Noise :    [%.2f]",SigmaNoise);
 printf("\n Threshold DCT :    [%.2f]",ThresholdDCT);
 printf("\n -----"); 
 printf("\n Zoom: %d",zoom); 
 printf("\n -----");
 printf("\n");

 
//-------------------------------
//---- Load Images -------------------------------
//-------------------------------

 //Information 
 GetLengthWidth(Name_img,&length,&width);
 printf("\n Taille de l'image: [%d - %d]",length,width);
 fflush(stdout);

 float** Img=fmatrix_allocate_2d(length,width); 
 float** ImgDegrad=fmatrix_allocate_2d(length,width); 

 load_image(Img,Name_img,length,width);
 copy_matrix(ImgDegrad,Img,length,width);
 
 if (SigmaNoise) 
 add_gaussian_noise(ImgDegrad,length,width,SigmaNoise*SigmaNoise);
 save_picture_pgm((char*)"",(char*)"DEGRADED",ImgDegrad,length,width);

//-------------------------------
//---- Denoising -------------------------------
//-------------------------------

 DCTDenoise* denoi;
 denoi=new DCTDenoise(ImgDegrad,length,width);
 (*denoi).Options(Img,SigmaNoise,ThresholdDCT);
    
  printf("\n\n  Info Simulated Noise");
  printf("\n  ---------------------");
  printf("\n    > MSE = [%.2f]\n\n",computeMMSE(ImgDegrad,Img,length));

  //DCT Denoising 
  (*denoi).IterDctDenoise();
 

//---------------------------------------------------------------
//---------------- vizualization  XWINDOW -----------------------
//---------------------------------------------------------------
 if (!flag_quit)
 {
 //Open Display
 if (open_display()<0) printf(" Impossible d'ouvrir une session graphique");

 //Comments
 sprintf(nomfen_ppicture_result,"Image Result : %s",Name_img); 
 sprintf(nomfen_ppicture_noise,"Image Noisy: %s",Name_img); 
 sprintf(nomfen_ppicture,"Image : %s",Name_img); 
 
 //Window creation 
 win_ppicture_result=fabrique_window(nomfen_ppicture_result,10,10,width,length,zoom);
 win_ppicture_noise=fabrique_window(nomfen_ppicture_noise,10,10,width,length,zoom);
 win_ppicture=fabrique_window(nomfen_ppicture,10,10,width,length,zoom);
 
 //Ximages creation
 x_ppicture_result=cree_Ximage2((*denoi).DataFiltered,zoom,length,width);
 x_ppicture_noise=cree_Ximage2(ImgDegrad,zoom,length,width); 
 x_ppicture=cree_Ximage2(Img,zoom,length,width);

 printf("\n\n Pour quitter,appuyer sur la barre d'espace");

 //Events loop
  for(;;)
     {
      XNextEvent(display,&ev);
       switch(ev.type)
        {
         case Expose:     
         XPutImage(display,win_ppicture_result,gc,x_ppicture_result,0,0,0,0,
                   x_ppicture_result->width,x_ppicture_result->height);
         XPutImage(display,win_ppicture_noise,gc,x_ppicture_noise,0,0,0,0,
                   x_ppicture_noise->width,x_ppicture_noise->height);
         XPutImage(display,win_ppicture,gc,x_ppicture,0,0,0,0,
                   x_ppicture->width,x_ppicture->height);
         break;

         case KeyPress:
         XDestroyImage(x_ppicture_result);
         XDestroyImage(x_ppicture_noise);
         XDestroyImage(x_ppicture);
         XFreeGC(display,gc);
         XCloseDisplay(display);
         flag=1;
         break;
         }
   if (flag==1) break;
   }
 } 
       
//--------------- End Graphical Session --------------------     
//----------------------------------------------------------
  
   //Return
   std::cout << "\n C'est fini... \n" << std::endl ; 
   return 0;
 }
 


