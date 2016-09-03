#ifndef _LOS_H_
#define _LOS_H_

#ifdef __cplusplus
extern "C" {
#endif

    #define LOS_VISIBLE ('.')
    #define LOS_BLOCKED ('@')
    //#define GPGPU(__f__,__t__) (__f__ ## __t__)    

    void initDevice();
    void killDevice(int argc, const char *argv);

    void losPointRectHOST(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt);
    void losPointPointsHOST(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt, unsigned int *pts, unsigned int n);
    void losPointsPointsHOST(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int *pts, unsigned int n);

    void losPointRectDEVICE(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt);
    void losPointPointsDEVICE(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int pt, unsigned int *pts, unsigned int n);
    void losPointsPointsDEVICE(unsigned char *los, float *hmap, unsigned int w, unsigned int h, unsigned int *pts, unsigned int n);
    
    void dumpMatrixF(const char *title, const char *fmt, float *matrix, unsigned int w, unsigned int h);
    void dumpMatrixLOS(const char *title, unsigned char *los, unsigned int w, unsigned int h, unsigned int viewpt);

    void timerCreate(unsigned int *timer);
    void timerStart(unsigned int timer);
    float timerStop(unsigned int timer);

#ifdef __cplusplus
}
#endif

#endif
