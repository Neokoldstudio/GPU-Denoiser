TEMPLATE	= app
CONFIG		= qt warn_on release -Wunused-but-set-variable 
HEADERS		= DCTFunction.h \
                  DCT-Denoise.h
SOURCES	        = DCTFunction.cc \
                  f_DCT-Denoise.cc\
                  DCT-Denoise.cc 
TARGET		=  DCT-Denoise
LIBS            = -lm -L/usr/X11R6/lib -lSM -lICE -lXext -lX11




