// local
extern "C"
#include <los.h>
#include <BMPLoader.h>
// ansi
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <cmath>


//=====================================
// CONSTANTS, MACROS, DEFINITIONS
//=====================================

#define DO_GPU          1       // calc gpu version
#define DO_CPU          0       // calc cpu version
#define PRINT_MATRIX    1
#define ITERATIONS      100
//#define HEIGHT_MAP      "0016.bmp"
#define HEIGHT_MAP      "0064.bmp"
//#define HEIGHT_MAP      "0128.bmp"
//#define HEIGHT_MAP      "0512.bmp"
//#define HEIGHT_MAP      "1024.bmp"
//#define HEIGHT_MAP      "2048.bmp"
//#define HEIGHT_MAP      "4096.bmp"

void testPointWholeMap();
void testLine();
void loadHeightmap();
void cleanup();

float *g_heightmap = 0;
unsigned int g_width = 0;
unsigned int g_height = 0;
unsigned int g_size = 0;
unsigned int g_timer = 0;


//=====================================
// TEST FUNCTIONS
//=====================================

int main()
{    
    loadHeightmap();
    initDevice();
    timerCreate(&g_timer);
    srand(time(0));

    //testLine();
    testPointWholeMap();
    printf("\n");

    //killDevice(argc,argv);
    cleanup();
    return 0;
}

void testPointWholeMap()
{
    unsigned char *los = new unsigned char[g_size];
    //unsigned int viewpt = 24;
    //unsigned int viewpt = 2025;
    //unsigned int viewpt = 7892;
    //unsigned int viewpt = 8145;
    //unsigned int viewpt = 128344; 
	unsigned int viewpt = 0;
    unsigned int i=0;
    float tgpu=0, tcpu=0;

#if DO_CPU
    memset(los,LOS_VISIBLE,g_size);
    viewpt = rand()%g_size;
    timerStart(g_timer);
    losPointRectHOST(los, g_heightmap, g_width,g_height, viewpt);
    tcpu = timerStop(g_timer);
	printf("CPU: %.2fs (%.2flps)\n", tcpu, g_size/tcpu);
#if PRINT_MATRIX
    dumpMatrixLOS("LOS: HOST\n", los, g_width,g_height, viewpt);
#endif
#endif

#if DO_GPU
    memset(los,LOS_VISIBLE,g_size);
    for (i=0; i<ITERATIONS; i++)
    {
        viewpt = rand()%g_size;
        timerStart(g_timer);
        losPointRectDEVICE(los, g_heightmap, g_width,g_height, viewpt);
        tgpu += timerStop(g_timer);
    }
	printf("GPU: %.4fs (%.4fs) (%.4flps)\n", tgpu, tgpu/(float)ITERATIONS, ((float)g_size)/(tgpu/(float)ITERATIONS));
#if PRINT_MATRIX
    dumpMatrixLOS("LOS: DEVICE\n", los, g_width,g_height, viewpt);   
#endif
#endif

    if (los)
    {
        delete [] los;
        los = 0;
    }
}

void testLine()
{
    int   ai = (g_size-1)-(g_width-1);
    float ax = ai % g_width;
    float ay = ai / g_width;
    float az = g_heightmap[ai];
    int   bi = g_width-1;
    float bx = bi % g_width;
    float by = bi / g_width;
    float bz = g_heightmap[bi];
    float cx;
    float cy;
    float t,step;
    unsigned char *los=0;
        
    step = 1.0f/sqrtf((double)(((bx-ax)*(bx-ax)) + ((by-ay)*(by-ay)) + ((bz-az)*(bz-az))));
    t    = step;
    los  = new unsigned char[g_size];    
    memset(los,LOS_VISIBLE,g_size);

    //dumpMatrixLOS("Empty:\n", los, g_width,g_height, ai);
    while (t<1.0f)
    {
        cx = (ax*(1.0f-t)) + (bx*t);
        cy = (ay*(1.0f-t)) + (by*t);
        los[(int)cy*g_width+(int)cx] = LOS_BLOCKED;
        t += step;
    }
    dumpMatrixLOS("Line:\n", los, g_width,g_height, ai);

    if (los)
    {
        delete [] los;
        los = 0;
    }
}

void loadHeightmap()
{
    // reset
    cleanup();    

    // load
    BMPLoader loader;
    loader.Load( HEIGHT_MAP );
    Pixel *pixels = loader.GetPixels();    

    // set dimensions
    g_width   = loader.GetWidth();
    g_height  = loader.GetHeight();
    g_size    = loader.GetNumPixels();

    // allocate memory (including pad)
    g_heightmap = new float[g_size];

    // extract heights    
    for (unsigned int i=0; i<g_size; i++)
        g_heightmap[i] = pixels[i].r;    
}

void cleanup()
{
    if (g_heightmap)
    {
        delete [] g_heightmap;
        g_heightmap = 0;
    }
}

