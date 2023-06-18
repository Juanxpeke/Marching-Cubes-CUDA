// ====================================================
// Code modified from NVIDIA CORPORATION, CUDA samples.
// By Juan Flores.
// ====================================================

#ifndef _MARCHING_CUBES_KERNEL_CU_
#define _MARCHING_CUBES_KERNEL_CU_

#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <helper_math.h>

#include "defines.h"
#include "tables.h"

// Textures containing look-up tables
cudaTextureObject_t triTex;
cudaTextureObject_t numVertsTex;

// ===================================
// ======== Density functions ========
// ===================================

__device__ float tangle(float x, float y, float z)
{
  x *= 3.0f;
  y *= 3.0f;
  z *= 3.0f;
  return (x * x * x * x - 5.0f * x * x + y * y * y * y - 5.0f * y * y +
          z * z * z * z - 5.0f * z * z + 11.8f) * 0.2f + 0.5f;
}

__device__ float wave(float x, float y, float z)
{
  float noise = (sin(x * 28.0f)  + cos(z * 28.0f)) * 0.036f;
  return (y + noise);
}

__device__ float sphere(float x, float y, float z)
{
  return sqrtf(x * x + y * y + z * z);
}

// ======================
// ======== Misc ========
// ======================

// Evaluate field function at point
__device__ float fieldFunc(float3 p) { return sphere(p.x, p.y, p.z); }

// Evaluate field function at a point, returns value and gradient in float4
__device__ float4 fieldFunc4(float3 p)
{
  float v = sphere(p.x, p.y, p.z);
  const float d = 0.001f;
  float dx = sphere(p.x + d, p.y, p.z) - v;
  float dy = sphere(p.x, p.y + d, p.z) - v;
  float dz = sphere(p.x, p.y, p.z + d) - v;
  return make_float4(dx, dy, dz, v);
}

// Compute position in 3D grid from 1d index, only works for power of 2 sizes
__device__ uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
  uint3 gridPos;
  gridPos.x = i & gridSizeMask.x;
  gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
  gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
  return gridPos;
}

// Calculate triangle normal
// NOTE: It's faster to perform normalization in vertex shader rather than here
__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
  float3 edge0 = *v1 - *v0;
  float3 edge1 = *v2 - *v0;
  return cross(edge0, edge1);
}

// Compute interpolated vertex along an edge
__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0,
                               float f1)
{
  float t = (isolevel - f0) / (f1 - f0);
  return lerp(p0, p1, t);
}

// Compute interpolated vertex position and normal along an edge
__device__ void vertexInterp2(float isolevel, float3 p0, float3 p1, float4 f0,
                              float4 f1, float3 &p, float3 &n)
{
  float t = (isolevel - f0.w) / (f1.w - f0.w);
  p = lerp(p0, p1, t);
  n.x = lerp(f0.x, f1.x, t);
  n.y = lerp(f0.y, f1.y, t);
  n.z = lerp(f0.z, f1.z, t);
  //    n = normalize(n);
}

// =========================
// ======== Kernels ========
// =========================

// Classify voxel based on number of vertices it will generate
// one thread per voxel
__global__ void classifyVoxel(
  uint *voxelVerts, uint *voxelOccupied, // Global data
  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, // Grid values
  uint numVoxels, float3 voxelSize, float isoValue, // More values
  cudaTextureObject_t numVertsTex) // CUDA texture
{
  uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

  uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

  float3 p;
  p.x = -0.5 * (gridSize.x * voxelSize.x) + (gridPos.x * voxelSize.x);
  p.y = -0.5 * (gridSize.y * voxelSize.y) + (gridPos.y * voxelSize.y);
  p.z = -0.5 * (gridSize.z * voxelSize.z) + (gridPos.z * voxelSize.z);

  float field[8];
  field[0] = fieldFunc(p);
  field[1] = fieldFunc(p + make_float3(voxelSize.x, 0, 0));
  field[2] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, 0));
  field[3] = fieldFunc(p + make_float3(0, voxelSize.y, 0));
  field[4] = fieldFunc(p + make_float3(0, 0, voxelSize.z));
  field[5] = fieldFunc(p + make_float3(voxelSize.x, 0, voxelSize.z));
  field[6] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z));
  field[7] = fieldFunc(p + make_float3(0, voxelSize.y, voxelSize.z));


  // Calculate flag indicating if each vertex is inside or outside isosurface
  uint cubeindex;
  cubeindex = uint(field[0] < isoValue);
  cubeindex += uint(field[1] < isoValue) * 2;
  cubeindex += uint(field[2] < isoValue) * 4;
  cubeindex += uint(field[3] < isoValue) * 8;
  cubeindex += uint(field[4] < isoValue) * 16;
  cubeindex += uint(field[5] < isoValue) * 32;
  cubeindex += uint(field[6] < isoValue) * 64;
  cubeindex += uint(field[7] < isoValue) * 128;

  // Read number of vertices from texture
  uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

  // TODO: Padding (?)
  if (i < numVoxels) {
    voxelVerts[i] = numVerts;
    voxelOccupied[i] = (numVerts > 0);
  }
}

// Generate triangles for each voxel using marching cubes, interpolates normals from field function
__global__ void generateTriangles(
  float4 *pos, float4 *norm, // VBO data
  uint *numVertsScanned, // Global data
  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, // Grid values
  float3 voxelSize, float isoValue, uint maxVerts, // More values
  cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex) // CUDA textures
{
  uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

  uint voxel = i;

  uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

  float3 p;
  p.x = -0.5 * (gridSize.x * voxelSize.x) + (gridPos.x * voxelSize.x);
  p.y = -0.5 * (gridSize.y * voxelSize.y) + (gridPos.y * voxelSize.y);
  p.z = -0.5 * (gridSize.z * voxelSize.z) + (gridPos.z * voxelSize.z);

  float3 v[8];
  v[0] = p;
  v[1] = p + make_float3(voxelSize.x, 0, 0);
  v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
  v[3] = p + make_float3(0, voxelSize.y, 0);
  v[4] = p + make_float3(0, 0, voxelSize.z);
  v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
  v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
  v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

  float4 field[8];
  field[0] = fieldFunc4(v[0]);
  field[1] = fieldFunc4(v[1]);
  field[2] = fieldFunc4(v[2]);
  field[3] = fieldFunc4(v[3]);
  field[4] = fieldFunc4(v[4]);
  field[5] = fieldFunc4(v[5]);
  field[6] = fieldFunc4(v[6]);
  field[7] = fieldFunc4(v[7]);

  // Recalculate flag (this is faster than storing it in global memory)
  uint cubeindex;
  cubeindex = uint(field[0].w < isoValue);
  cubeindex += uint(field[1].w < isoValue) * 2;
  cubeindex += uint(field[2].w < isoValue) * 4;
  cubeindex += uint(field[3].w < isoValue) * 8;
  cubeindex += uint(field[4].w < isoValue) * 16;
  cubeindex += uint(field[5].w < isoValue) * 32;
  cubeindex += uint(field[6].w < isoValue) * 64;
  cubeindex += uint(field[7].w < isoValue) * 128;

// Find the vertices where the surface intersects the cube

#if USE_SHARED
  // Use partioned shared memory to avoid using local memory
  __shared__ float3 vertlist[12 * NTHREADS];
  __shared__ float3 normlist[12 * NTHREADS];

  vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[threadIdx.x],
                normlist[threadIdx.x]);
  vertexInterp2(isoValue, v[1], v[2], field[1], field[2],
                vertlist[threadIdx.x + NTHREADS],
                normlist[threadIdx.x + NTHREADS]);
  vertexInterp2(isoValue, v[2], v[3], field[2], field[3],
                vertlist[threadIdx.x + (NTHREADS * 2)],
                normlist[threadIdx.x + (NTHREADS * 2)]);
  vertexInterp2(isoValue, v[3], v[0], field[3], field[0],
                vertlist[threadIdx.x + (NTHREADS * 3)],
                normlist[threadIdx.x + (NTHREADS * 3)]);
  vertexInterp2(isoValue, v[4], v[5], field[4], field[5],
                vertlist[threadIdx.x + (NTHREADS * 4)],
                normlist[threadIdx.x + (NTHREADS * 4)]);
  vertexInterp2(isoValue, v[5], v[6], field[5], field[6],
                vertlist[threadIdx.x + (NTHREADS * 5)],
                normlist[threadIdx.x + (NTHREADS * 5)]);
  vertexInterp2(isoValue, v[6], v[7], field[6], field[7],
                vertlist[threadIdx.x + (NTHREADS * 6)],
                normlist[threadIdx.x + (NTHREADS * 6)]);
  vertexInterp2(isoValue, v[7], v[4], field[7], field[4],
                vertlist[threadIdx.x + (NTHREADS * 7)],
                normlist[threadIdx.x + (NTHREADS * 7)]);
  vertexInterp2(isoValue, v[0], v[4], field[0], field[4],
                vertlist[threadIdx.x + (NTHREADS * 8)],
                normlist[threadIdx.x + (NTHREADS * 8)]);
  vertexInterp2(isoValue, v[1], v[5], field[1], field[5],
                vertlist[threadIdx.x + (NTHREADS * 9)],
                normlist[threadIdx.x + (NTHREADS * 9)]);
  vertexInterp2(isoValue, v[2], v[6], field[2], field[6],
                vertlist[threadIdx.x + (NTHREADS * 10)],
                normlist[threadIdx.x + (NTHREADS * 10)]);
  vertexInterp2(isoValue, v[3], v[7], field[3], field[7],
                vertlist[threadIdx.x + (NTHREADS * 11)],
                normlist[threadIdx.x + (NTHREADS * 11)]);
  __syncthreads();

#else
  float3 vertlist[12];
  float3 normlist[12];

  vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[0],
                normlist[0]);
  vertexInterp2(isoValue, v[1], v[2], field[1], field[2], vertlist[1],
                normlist[1]);
  vertexInterp2(isoValue, v[2], v[3], field[2], field[3], vertlist[2],
                normlist[2]);
  vertexInterp2(isoValue, v[3], v[0], field[3], field[0], vertlist[3],
                normlist[3]);

  vertexInterp2(isoValue, v[4], v[5], field[4], field[5], vertlist[4],
                normlist[4]);
  vertexInterp2(isoValue, v[5], v[6], field[5], field[6], vertlist[5],
                normlist[5]);
  vertexInterp2(isoValue, v[6], v[7], field[6], field[7], vertlist[6],
                normlist[6]);
  vertexInterp2(isoValue, v[7], v[4], field[7], field[4], vertlist[7],
                normlist[7]);

  vertexInterp2(isoValue, v[0], v[4], field[0], field[4], vertlist[8],
                normlist[8]);
  vertexInterp2(isoValue, v[1], v[5], field[1], field[5], vertlist[9],
                normlist[9]);
  vertexInterp2(isoValue, v[2], v[6], field[2], field[6], vertlist[10],
                normlist[10]);
  vertexInterp2(isoValue, v[3], v[7], field[3], field[7], vertlist[11],
                normlist[11]);
#endif

  // Output triangle vertices
  uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

  for (int i = 0; i < numVerts; i++) {
    uint edge = tex1Dfetch<uint>(triTex, cubeindex * 16 + i);

    uint index = numVertsScanned[voxel] + i;

    if (index < maxVerts) {
#if USE_SHARED
      pos[index] = make_float4(vertlist[(edge * NTHREADS) + threadIdx.x], 1.0f);
      norm[index] = make_float4(normlist[(edge * NTHREADS) + threadIdx.x], 0.0f);
#else
      pos[index] = make_float4(vertlist[edge], 1.0f);
      norm[index] = make_float4(normlist[edge], 0.0f);
#endif
    }
  }
}

// ========================
// ======== Extern ========
// ========================

extern "C" void launchClassifyVoxel(
  dim3 grid, dim3 threads, 
  uint *voxelVerts, uint *voxelOccupied,
  uint3 gridSize, uint3 gridSizeShift,
  uint3 gridSizeMask, uint numVoxels,
  float3 voxelSize, float isoValue)
{
  classifyVoxel<<<grid, threads>>>(
  voxelVerts, voxelOccupied,
  gridSize, gridSizeShift, gridSizeMask,
  numVoxels, voxelSize, isoValue,
  numVertsTex);
}

extern "C" void launchGenerateTriangles(
  dim3 grid, dim3 threads,
  float4 *pos, float4 *norm,
  uint *numVertsScanned,
  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
  float3 voxelSize, float isoValue, uint maxVerts)
{
  generateTriangles<<<grid, NTHREADS>>>(
    pos, norm, 
    numVertsScanned,
    gridSize, gridSizeShift, gridSizeMask,
    voxelSize, isoValue, maxVerts,
    triTex, numVertsTex);
}

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
                                  unsigned int numElements)
{
  thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
                         thrust::device_ptr<unsigned int>(input + numElements),
                         thrust::device_ptr<unsigned int>(output));
}

extern "C" void allocateTextures(uint **dEdgeTable, uint **dTriTable, uint **dNumVertsTable)
{
  cudaMalloc((void**) dEdgeTable, 256 * sizeof(uint));
  cudaMemcpy((void*) *dEdgeTable, (void*) edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaMalloc((void**) dTriTable, 256 * 16 * sizeof(uint));
  cudaMemcpy((void*) *dTriTable, (void*) triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice);

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = *dTriTable;
  texRes.res.linear.sizeInBytes = 256 * 16 * sizeof(uint);
  texRes.res.linear.desc = channelDesc;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&triTex, &texRes, &texDescr, NULL);

  cudaMalloc((void **)dNumVertsTable, 256 * sizeof(uint));
  cudaMemcpy((void *)*dNumVertsTable, (void *)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice);

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = *dNumVertsTable;
  texRes.res.linear.sizeInBytes = 256 * sizeof(uint);
  texRes.res.linear.desc = channelDesc;

  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&numVertsTex, &texRes, &texDescr, NULL);
}

extern "C" void destroyAllTextureObjects() {
  cudaDestroyTextureObject(triTex);
  cudaDestroyTextureObject(numVertsTex);
}

#endif
