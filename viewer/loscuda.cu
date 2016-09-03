// local
#include <los.h>
// ansi
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// cuda
#include <cutil.h>


//=====================================
// CONSTANTS, MACROS, DEFINITIONS
//=====================================

#define BLOCK_SIZE  16


//========================================
// DEVICE FUNCTIONS
//========================================

__device__ float _sampleBicubicFilter(float *hmap, unsigned int w, unsigned int h, float u, float v)
{
    unsigned int baseu, basev;
    float heights[6];

    // get the base index for the 2x2 block
    baseu = floorf(u);
    basev = floorf(v);

    // sample the 2x2 texel block
    heights[0] = hmap[basev*w+baseu];
    heights[1] = hmap[basev*w+min(baseu+1,w-1)];
    heights[2] = hmap[min(basev+1,h-1)*w+baseu];
    heights[3] = hmap[min(basev+1,h-1)*w+min(baseu+1,w-1)];

    // lerp along the horizontal (2 rows)
    heights[4] = heights[0] + (heights[1] - heights[0]) * (u - baseu);  // lerp 0 & 1
    heights[5] = heights[2] + (heights[3] - heights[2]) * (u - baseu);  // lerp 2 & 3
    return heights[4] + (heights[5] - heights[4]) * (v - basev);        // lerp results above
}

__global__ void _losPointRectDEVICE(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt)
{
    int x  = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y  = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int bi = y * w + x;
    float ax,ay,az;
    float bx,by,bz;
    float cx,cy,cz;
    float length, sample;
    float i,t;

    // point A
    ax = pt % w;
    ay = pt / w;
    az = hmap[pt];

    // point B
    bx = x;
    by = y;
    bz = hmap[bi];

    // line AB
    length = sqrtf( ((bx-ax)*(bx-ax)) + ((by-ay)*(by-ay)) + ((bz-az)*(bz-az)) );

    // los algorithm: sample hmap along line
    i=1.0f; t=0;
    while (i<length)
    {
        t = i / length;
        cx = (ax*(1.0f-t)) + (bx*t);
        cy = (ay*(1.0f-t)) + (by*t);
        cz = (az*(1.0f-t)) + (bz*t);
        sample = _sampleBicubicFilter(hmap,w,h,cx,cy);

        if (sample-cz>1.0f) // use function 'signbit()'
        {
            los[bi]=LOS_BLOCKED;
            break;
        }

        i+=0.5f;
    }     
}


//=====================================
// HOST FUNCTIONS
//=====================================

__host__ void initDevice()
{
    CUT_DEVICE_INIT();
}

__host__ void killDevice(int argc, const char *argv)
{
    CUT_EXIT(argc, argv);
}

__host__ void timerCreate(unsigned int *timer)
{
    CUT_SAFE_CALL(cutCreateTimer(timer));
}

__host__ void timerStart(unsigned int timer)
{
    CUT_SAFE_CALL(cutResetTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
}

__host__ float timerStop(unsigned int timer)
{
    CUT_SAFE_CALL(cutStopTimer(timer));
    return (cutGetTimerValue(timer) / 1000.0f);
}

__host__ void losPointRectDEVICE(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt)
{
    unsigned int HMAP_SIZE = w*h;
    unsigned char  *d_los;  CUDA_SAFE_CALL(cudaMalloc((void**)&d_los,  HMAP_SIZE*sizeof(unsigned char)));
    float          *d_hmap; CUDA_SAFE_CALL(cudaMalloc((void**)&d_hmap, HMAP_SIZE*sizeof(float)));

    CUDA_SAFE_CALL(cudaMemcpy(d_los,  los,  HMAP_SIZE*sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_hmap, hmap, HMAP_SIZE*sizeof(float),         cudaMemcpyHostToDevice));    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(w/BLOCK_SIZE, h/BLOCK_SIZE);
    _losPointRectDEVICE<<<blocks, threads>>>(d_los, d_hmap, w, h, pt);
    CUT_CHECK_ERROR("Kernel execution failed!");
    CUDA_SAFE_CALL(cudaMemcpy(los, d_los, HMAP_SIZE*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_los));
    CUDA_SAFE_CALL(cudaFree(d_hmap));
}
