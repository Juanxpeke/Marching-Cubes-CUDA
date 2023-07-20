// ====================================================
// Code modified from NVIDIA CORPORATION, CUDA samples.
// By Juan Flores.
// ====================================================

#ifndef _DEFINES_H_
#define _DEFINES_H_

typedef unsigned int uint;
typedef unsigned char uchar;

// Using shared to store computed vertices and normals during triangle
// generation improves performance
#define USE_SHARED 1

// The number of threads to use for triangle generation (limited by shared
// memory size)
#define NTHREADS 32

// OpenGL
#define DISABLE_FPS_CAPPING 1
#define ENABLE_FACE_CULLING 1

#endif
