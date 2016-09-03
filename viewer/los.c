// local
#include "los.h"
// ansi
#include <stdio.h>
#include <math.h>



//=====================================
// MACROS, CONSTANTS, DEFINITIONS
//=====================================

float sampleBicubicFilter(float *hmap, unsigned int w, unsigned int h, float u, float v);


//=====================================
// HOST FUNCTIONS
//=====================================

void losPointRectHOST(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt)
{    
    float ax,ay,az;
    float bx,by,bz;
    float cx,cy,cz;
    float length, sample;
    float i,t;
    unsigned int bi=0;

    // point A
    ax = pt % w;
    ay = pt / w;
    az = hmap[pt];

    // loop over all points in rectangle
    for (by=0; by<h; by++)
    {
        for (bx=0; bx<w; bx++)
        {
            // point B
            // bx = j
            // by = i
            bz = hmap[bi];

            // line AB
            length = sqrt( ((bx-ax)*(bx-ax)) + ((by-ay)*(by-ay)) + ((bz-az)*(bz-az)) );

            // los algorithm: sample hmap along line
            i=1.0f; t=0;
            while (i<length)
            {
                t = i / length;
                cx = (ax*(1.0f-t)) + (bx*t);
                cy = (ay*(1.0f-t)) + (by*t);
                cz = (az*(1.0f-t)) + (bz*t);
                //ci = (int)cy*w+(int)cx;
                sample = sampleBicubicFilter(hmap,w,h,cx,cy);

                if (sample-cz>1.0f)
                {
                    los[bi]=LOS_BLOCKED;
                    break;
                }

                i+=0.5f;
            }

            bi++;
        }
    }
}


//=====================================
// UTILITY FUNCTIONS
//=====================================

float sampleBicubicFilter(float *hmap, unsigned int w, unsigned int h, float u, float v)
{
    unsigned int baseu, basev, basei;
    float heights[6];

    // get the base index for the 2x2 block
    baseu = floor(u);
    basev = floor(v);

    // sample the 2x2 texel block
    basei = basev*w+baseu;
    heights[0] = hmap[basei];
    if (baseu==w)           heights[1] = hmap[basei]; else heights[1] = hmap[basev*w+baseu+1];
    if (basev==h)           heights[2] = hmap[basei]; else heights[2] = hmap[(basev+1)*w+baseu];
    if (baseu==w&&basev==h) heights[3] = hmap[basei]; else heights[3] = hmap[(basev+1)*w+baseu+1];

    // lerp along the horizontal (2 rows) 
    heights[4] = heights[0] + (heights[1] - heights[0]) * (u - baseu);  // lerp 0 & 1
    heights[5] = heights[2] + (heights[3] - heights[2]) * (u - baseu);  // lerp 2 & 3
    return heights[4] + (heights[5] - heights[4]) * (v - basev);        // lerp results above
}


void dumpMatrixF(const char *title, const char *fmt, float *matrix, unsigned int w, unsigned int h)
{
    unsigned int i,j;
    printf(title);
    for (j=0; j<h; j++)
    {
        for (i=0; i<w; i++)
            printf( fmt, matrix[j*w+i] );
        printf( "\n" );
    }
    printf( "\n" );
}

void dumpMatrixLOS(const char *title, unsigned char *los, unsigned int w, unsigned int h, unsigned int viewpt)
{
    unsigned int i,j;
    los[viewpt]='^';
    printf(title);
    for (j=0; j<h; j++)
    {
        for (i=0; i<w; i++)
            printf( "%c", los[j*w+i] );
        printf( "\n" );
    }
    printf( "\n" );
}
