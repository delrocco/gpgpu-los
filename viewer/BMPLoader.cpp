//=============================================================================
//=============================================================================
// BMPLoader.cpp
// 
// Class for loading .bmp images.
//=============================================================================
//=============================================================================

// local
#include "BMPLoader.h"
// ansi
#include <stdio.h>
#include <memory.h>


//=====================================
// CONSTANTS, MACROS, DEFINITIONS
//=====================================


//=====================================
// PUBLIC FUNCTIONS
//=====================================

BMPLoader::BMPLoader()
{
    colors = 0;
    pixels = 0;
    memset( &fileheader, 0, sizeof(BMPFileHeader) );
    memset( &infoheader, 0, sizeof(BMPInfoHeader) );
}

BMPLoader::~BMPLoader()
{
    reset();
}

int BMPLoader::Load( const char *image )
{    
    FILE *file=0;
    unsigned char *buffer=0;

    // quick out
    if (!image) return kFailBadPath;

    // clean loader
    reset();

    // open file
    file = fopen( image, "r" );
    if (!file) return kFailBadPath;

    // load file header
    //fread( &fileheader, sizeof(BMPFileHeader), 1, file ); // <- BAD IDEA
    fread( &fileheader.bfType,      sizeof(unsigned short), 1, file );
    fread( &fileheader.bfSize,      sizeof(unsigned long),  1, file );
    fread( &fileheader.bfReserved1, sizeof(unsigned short), 1, file );
    fread( &fileheader.bfReserved2, sizeof(unsigned short), 1, file );
    fread( &fileheader.bfOffBits,   sizeof(unsigned long),  1, file );

    // load info header
    //fread( &infoheader, sizeof(BMPInfoHeader), 1, file ); // <- BAD IDEA
    fread( &infoheader.biSize,          sizeof(unsigned long), 1, file );
    fread( &infoheader.biWidth,         sizeof(long), 1, file );
    fread( &infoheader.biHeight,        sizeof(long), 1, file );
    fread( &infoheader.biPlanes,        sizeof(unsigned short), 1, file );
    fread( &infoheader.biBitCount,      sizeof(unsigned short), 1, file );
    fread( &infoheader.biCompression,   sizeof(unsigned long), 1, file );
    fread( &infoheader.biSizeImage,     sizeof(unsigned long), 1, file );
    fread( &infoheader.biXPelsPerMeter, sizeof(long), 1, file );
    fread( &infoheader.biYPelsPerMeter, sizeof(long), 1, file );
    fread( &infoheader.biClrUsed,       sizeof(unsigned long), 1, file );
    fread( &infoheader.biClrImportant,  sizeof(unsigned long), 1, file );

    // TODO: handle 1,4,8 bit depth (color palettes)
    // TODO: handle 16 bit depth
    // TODO: handle compressed images

    // allocate appropriate memory
    long size = (fileheader.bfSize - fileheader.bfOffBits);    
    buffer = new unsigned char[size];
    pixels = new Pixel[infoheader.biWidth*infoheader.biHeight];

    // load pixel data
    fseek( file, fileheader.bfOffBits, SEEK_SET );
    fread( buffer, sizeof(unsigned char), size, file );

    // cleanup
    fclose(file);

    // process pixel data
    int bytespsl = ((((infoheader.biWidth*infoheader.biBitCount)+31)>>5)<<2);
    int bytespad = bytespsl - (((infoheader.biWidth*infoheader.biBitCount)+7)>>3);
    unsigned char *pixelbyte=buffer;
    unsigned int idx=0;
    for ( int j=infoheader.biHeight-1; j>=0; j-- )
    {        
        for ( int i=0; i<infoheader.biWidth; i++ )
        {
            idx = j*infoheader.biWidth+i;
            pixels[idx].b = *pixelbyte; pixelbyte++;
            pixels[idx].g = *pixelbyte; pixelbyte++;
            pixels[idx].r = *pixelbyte; pixelbyte++;
            pixels[idx].a = 0;      
        }
        pixelbyte+=bytespad;
    }

    // cleanup
    delete [] buffer;

    /*
    WINDOWS GDI loading method..

    BITMAP bitmap;
    HBITMAP hbitmap = (HBITMAP)LoadImage(NULL, image, IMAGE_BITMAP, 0,0, LR_CREATEDIBSECTION | LR_LOADFROMFILE);    
    GetObject(hbitmap, sizeof(bitmap), &bitmap);
    pixels = new Pixel[bitmap.bmWidth*bitmap.bmHeight];
    unsigned char *pixelbyte = (unsigned char*)bitmap.bmBits;
    unsigned int idx=0;
    for ( int j=bitmap.bmHeight; j>=0; j-- )
    {        
        for ( int i=0; i<bitmap.bmWidth; i++ )
        {
            pixels[idx].b = *pixelbyte; pixelbyte++;
            pixels[idx].g = *pixelbyte; pixelbyte++;
            pixels[idx].r = *pixelbyte; pixelbyte++;
        }
    }
    DeleteObject(hbitmap);
    */

    return kSuccess;
}

void BMPLoader::Cleanup()
{
    reset();
}

bool BMPLoader::IsLoaded()
{
    return (pixels != 0);
}

bool BMPLoader::IsCompressed()
{
    return (infoheader.biCompression != 0);
}

int BMPLoader::GetWidth()
{
    return infoheader.biWidth;
}

int BMPLoader::GetHeight()
{
    return infoheader.biHeight;
}

int BMPLoader::GetNumPixels()
{
    return (infoheader.biWidth * infoheader.biHeight);
}

int BMPLoader::GetPixelDepthBits()
{
    return infoheader.biBitCount;
}

int BMPLoader::GetPixelDepthBytes()
{
    return (infoheader.biBitCount>>3);
}

Pixel *BMPLoader::GetPixels()
{
    return pixels;
}

Pixel *BMPLoader::GetPixel( unsigned int idx )
{    
    return &pixels[idx];
}

Pixel *BMPLoader::GetPixel( unsigned int x, unsigned int y )
{    
    return &pixels[y*infoheader.biWidth+x];
}

void BMPLoader::Dump( const char *file )
{
    // quick outs
    if (!file)   return;
    if (!pixels) return;

    FILE *dump=0;
    unsigned int idx=0;
    dump = fopen(file, "w");
    for ( int j=0; j<infoheader.biHeight; j++ )
    {
        for ( int i=0; i<infoheader.biWidth; i++ )
        {
            //fprintf(dump, "[%2x,%2x,%2x]", pixels[idx].r, pixels[idx].g, pixels[idx].b );
            fprintf(dump, "[%2x]", pixels[idx].r );
            idx++;
        }
        fprintf(dump,"\n");
    }
    fclose(dump);
}


//=====================================
// PRIVATE FUNCTIONS
//=====================================

void BMPLoader::reset()
{
    if ( colors )
    {
        delete [] colors;
        colors = 0;
    }

    if ( pixels )
    {
        delete [] pixels;
        pixels = 0;
    }
}
