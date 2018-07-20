#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <vector_functions.h>
#include "isosurfaces_cuda/hdr/common.h"
#include <cmath>
#include "runtime/hdr/globals.h"


using namespace std;

#define MAX_PER_RUN 2000000

__device__ float float3SquaredDistance_d(float3 v1, float3 v2)
{
  float x2 = (v2.x - v1.x) * (v2.x - v1.x);
  float y2 = (v2.y - v1.y) * (v2.y - v1.y);
  float z2 = (v2.z - v1.z) * (v2.z - v1.z);
  return (x2 + y2 + z2);
}

__device__ float calculateMinimum_d(float3 samplePoint, float3 points[], const int num_points, const int offset, float min)
{

  float distance;
  for (int i = offset; i < offset+MAX_PER_RUN && i < num_points; ++i)
  {
    distance = float3SquaredDistance_d(points[i], samplePoint);
    if (distance < min)
    {
      min = distance;
    }
  }
  return min;
}

__device__ float3 findSamplePointForIndex(int idx, int dimY, int dimZ, int dfDownscale)
{
  int z = idx % dimZ;
  int y = (idx / dimZ) % dimY;
  int x = idx / (dimY * dimZ);

  return make_float3(x*dfDownscale+dfDownscale/2, y*dfDownscale+dfDownscale/2, z*dfDownscale+dfDownscale/2);

}

__global__ void calculate_distances_g(float3 points[], const int num_points, int dimY, int dimZ, int dfDownscale, float *distances, int size, const int offset)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
  {
    float min;
    if (offset == 0) {
       min = 1e38;
    } else {
      min = distances[idx];
      min = min * min;
    }
    float3 pt = findSamplePointForIndex(idx, dimY, dimZ, dfDownscale);
    min = calculateMinimum_d(pt, points, num_points, offset, min);
    distances[idx] = sqrtf(min);
  }
}

extern "C"
__host__ void calculate_distances_h(vector<float3> points, int dimX, int dimY, int dimZ, int dfDownscale, float *h_distances)
{
  unsigned long SIZE = (unsigned long) (dimX * dimY * dimZ);
  unsigned long POINTS_SIZE = points.size() * sizeof(float3);
  unsigned long OUT_SIZE = SIZE * sizeof(float);

  float3 *d_points;
  float *d_distances;

  cudaMalloc((void **) &d_points, POINTS_SIZE);
  cudaMalloc((void **) &d_distances, OUT_SIZE);

  printf("numPoint: %lu\n", points.size());

  cudaMemcpy(d_points, &points[0], POINTS_SIZE, cudaMemcpyHostToDevice);
  for (int offset = 0; offset < points.size(); offset+=MAX_PER_RUN)
  {
    calculate_distances_g <<< calculateBlockNumber(SIZE, 256), 256 >>>(d_points, points.size(), dimY, dimZ, dfDownscale, d_distances, SIZE, offset);
  }
  cudaMemcpy(h_distances, d_distances, OUT_SIZE, cudaMemcpyDeviceToHost);
  printf("DISTANCEFIELD: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaFree(d_points);
  cudaFree(d_distances);

  return;

}
