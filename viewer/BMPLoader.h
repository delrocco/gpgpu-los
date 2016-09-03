//=============================================================================
//=============================================================================
// BMPLoader.h
// 
// Class for loading .bmp images.
//=============================================================================
//=============================================================================
#ifndef _BMPLOADER_H_
#define _BMPLOADER_H_


//=====================================
// CONSTANTS, MACROS, DEFINITIONS
//=====================================

typedef struct _BMPFileHeader
{
    unsigned short  bfType;
    unsigned long   bfSize;
    unsigned short  bfReserved1;
    unsigned short  bfReserved2;
    unsigned long   bfOffBits;
} BMPFileHeader;

typedef struct _BMPInfoHeader
{
    unsigned long   biSize;
    long            biWidth;
    long            biHeight;
    unsigned short  biPlanes;
    unsigned short  biBitCount;
    unsigned long   biCompression;
    unsigned long   biSizeImage;
    long            biXPelsPerMeter;
    long            biYPelsPerMeter;
    unsigned long   biClrUsed;
    unsigned long   biClrImportant;
} BMPInfoHeader;

typedef struct _BMPColor
{
    unsigned char   rgbBlue;
    unsigned char   rgbGreen;
    unsigned char   rgbRed;
    unsigned char   rgbReserved;
} BMPColor;

typedef struct _Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} Pixel;


//=====================================
// CLASS
//=====================================

class BMPLoader
{
public:
    enum ErrorCode
    {
        kSuccess        = 0x01,
        kFailBadPath    = 0x02,
        kFailBadHeader  = 0x04,
        kFailLessPixels = 0x08,
    };    

public:
    BMPLoader();
    ~BMPLoader();

    int Load( const char *file );
    void Cleanup();

    bool IsLoaded();
    bool IsCompressed();

    int GetWidth();
    int GetHeight();
    int GetNumPixels();
    int GetPixelDepthBits();
    int GetPixelDepthBytes();
    Pixel *GetPixels();
    Pixel *GetPixel( unsigned int idx );
    Pixel *GetPixel( unsigned int x, unsigned int y );

    void Dump( const char *file );

private:
    void reset();

private:
    BMPFileHeader   fileheader;
    BMPInfoHeader   infoheader;
    BMPColor       *colors;
    Pixel          *pixels;
};

#endif
