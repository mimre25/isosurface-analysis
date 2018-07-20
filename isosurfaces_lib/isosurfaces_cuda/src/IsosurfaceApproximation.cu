#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <vector_functions.h>
#include <cstring>
#include "isosurfaces_cuda/hdr/common.h"
#include "isosurfaces_cuda/hdr/thrustWrapper.h"
#include "isosurfaces_cuda/hdr/helpers.h"

using namespace std;


__host__ __device__ bool checkInside(int3 leftUpper, float isovalue, float* data, const int dimX, const int dimY, const int dimZ, const int scale)
{
  int3 IOO = make_int3(scale,0,0);
  int3 OIO = make_int3(0,scale,0);
  int3 OOI = make_int3(0,0,scale);
  int3 IIO = make_int3(scale,scale,0);
  int3 IOI = make_int3(scale,0,scale);
  int3 OII = make_int3(0,scale,scale);
  int3 III = make_int3(scale,scale,scale);

  bool above = false;
  bool below = false;

  float ooo = data[findPointIndex(leftUpper, dimX, dimY, dimZ)]; // should be idx
  above = ooo > isovalue;
  below = ooo < isovalue;
  float ioo = data[findPointIndex(leftUpper+IOO, dimX, dimY, dimZ)];
  above = above || ioo > isovalue;
  below = below || ioo < isovalue;
//  if (!above  || !below) {
    float oio = data[findPointIndex(leftUpper+OIO, dimX, dimY, dimZ)];
    above = above || oio > isovalue;
    below = below || oio < isovalue;
//  }
//  if (!above  || !below) {
    float ooi = data[findPointIndex(leftUpper+OOI, dimX, dimY, dimZ)];
    above = above || ooi > isovalue;
    below = below || ooi < isovalue;
  // }
//  if (!above  || !below) {
    float iio = data[findPointIndex(leftUpper+IIO, dimX, dimY, dimZ)];
    above = above || iio > isovalue;
    below = below || iio < isovalue;
//  }
//  if (!above  || !below) {
    float ioi = data[findPointIndex(leftUpper+IOI, dimX, dimY, dimZ)];
    above = above || ioi > isovalue;
    below = below || ioi < isovalue;
//  }
//  if (!above  || !below) {
    float oii = data[findPointIndex(leftUpper+OII, dimX, dimY, dimZ)];
    above = above || oii > isovalue;
    below = below || oii < isovalue;
//  }
  // if (!above  || !below) {
    float iii = data[findPointIndex(leftUpper+III, dimX, dimY, dimZ)];
    above = above || iii > isovalue;
    below = below || iii < isovalue;
  // }
  return above && below;
}

__host__ __device__ inline bool checkInside(int lu, float isovalue, float* data, const int dimX, const int dimY, const int dimZ, const int scale)
{
  int dimZY = scale*dimZ*dimY;
  int id1 = lu+scale;
  int id2 = lu+scale*dimZ;
  int id3 = id1+scale*dimZ;
  int id4 = lu+dimZY;
  int id5 = id1+dimZY;
  int id6 = id2+dimZY;
  int id7 = id3+dimZY;

  bool above = false;
  bool below = false;

  float ooo = data[lu]; // should be idx
  above = ooo > isovalue;
  below = ooo < isovalue;
  float ioo = data[id1];
  above = above || ioo > isovalue;
  below = below || ioo < isovalue;
  //if (!above  || !below) {
  float oio = data[id2];
  above = above || oio > isovalue;
  below = below || oio < isovalue;
  // }
  //if (!above  || !below) {
  float ooi = data[id3];
  above = above || ooi > isovalue;
  below = below || ooi < isovalue;
  // }
  //if (!above  || !below) {
  float iio = data[id4];
  above = above || iio > isovalue;
  below = below || iio < isovalue;
  // }
  // if (!above  || !below) {
  float ioi = data[id5];
  above = above || ioi > isovalue;
  below = below || ioi < isovalue;
  // }
  // if (!above  || !below) {
  float oii = data[id6];
  above = above || oii > isovalue;
  below = below || oii < isovalue;
  // }
  //  if (!above  || !below) {
  float iii = data[id7];
  above = above || iii > isovalue;
  below = below || iii < isovalue;
  // }
  return above && below;
}

__global__ void calculate_approximation_g(float* data, float isovalue, int dimX, int dimY, int dimZ, int3 *surfaceApprox, const int size, const int offset, const int scale)
{
  int idx = threadIdx.x + (blockIdx.x+offset) * blockDim.x;
  idx *= scale;
  if (idx < size)
  {

      int3 lu = findPointForIndex(idx, dimY, dimZ);
      if (lu.x < (dimX-scale) && lu.y < (dimY-scale) && lu.z < (dimZ-scale))
      {
        surfaceApprox[findOutputIndex(lu, dimX, dimY, dimZ, scale)] = checkInside(idx, isovalue, data, dimX, dimY, dimZ, scale) ? lu : make_int3(-1, -1, -1);
      }


  }
}


void calculate_approximation_CPU(float* data, float isovalue, int dimX, int dimY, int dimZ, int3 *surfaceApprox, const int size, const int offset, const int scale)
{
  int idx = 0;
  for (int i = 0; idx < size; ++i)
  {
    idx = i;
    idx *= scale;
    if (idx < size)
    {
      int3 lu = findPointForIndex(idx, dimY, dimZ);
      if (lu.x < (dimX-scale) && lu.y < (dimY-scale) && lu.z < (dimZ-scale))
      {
        surfaceApprox[findOutputIndex(lu, dimX, dimY, dimZ, scale)] = checkInside(idx, isovalue, data, dimX, dimY, dimZ, scale) ? lu : make_int3(-1,-1,-1);
      }
    }
  }


}


extern "C"
__host__ long approximate_isosurfaces_h(float* h_data, float isovalue, const int dimX, const int dimY, const int dimZ, const int scale, int3 **h_points)
{
  unsigned int NUM_THREADS = 256;
  unsigned int MAX_NUM_BLOCKS = (2<<14);

  unsigned long SIZE = (unsigned long) (dimX * dimY * dimZ);
  unsigned long POINTS_SIZE = SIZE * sizeof(float);
  unsigned long NUM_POINTS = ((dimX-scale)*(dimY-scale)*(dimZ-scale))/scale;
  unsigned long OUT_SIZE = NUM_POINTS * sizeof(int3);

  float *d_data;
  int3 *d_points;
  int3 *d_points2;

  cudaMalloc((void **) &d_points, OUT_SIZE);
  cudaMemset(d_points, 0xff, OUT_SIZE);
  cudaMalloc((void **) &d_data, POINTS_SIZE);

  cudaMemcpy(d_data, h_data, POINTS_SIZE, cudaMemcpyHostToDevice);

  int blockNumber = calculateBlockNumber(SIZE, NUM_THREADS);
  for (int offset = 0; offset < blockNumber; offset += MAX_NUM_BLOCKS)
  {
    calculate_approximation_g <<< MAX_NUM_BLOCKS, NUM_THREADS >>>(d_data, isovalue, dimX, dimY, dimZ, d_points, SIZE, offset, scale);
  }

  Reporter r;
  r.reportStart((char *) "compacting GPU");


  cudaMalloc((void **) &d_points2, OUT_SIZE);

  printf("APPROX: %s\n", cudaGetErrorString(cudaGetLastError()));


  cudaMemset(d_points2, 0, OUT_SIZE);

  long length = compact(d_points, d_points2, NUM_POINTS, is_positive());


  r.reportEnd();

  *h_points = (int3*) malloc(length * sizeof(int3));
  cudaMemcpy(*h_points, d_points2, length*sizeof(int3), cudaMemcpyDeviceToHost);
  cudaFree(d_points2);


  cudaFree(d_data);
  cudaFree(d_points);

  return length;

}

extern "C"
__host__ long approximate_isosurfaces_CPU_h(float* h_data, float isovalue, const int dimX, const int dimY, const int dimZ, const int scale, int3 *h_points)
{

  unsigned long SIZE = (unsigned long) (dimX * dimY * dimZ);

  calculate_approximation_CPU (h_data, isovalue, dimX, dimY, dimZ, h_points, SIZE, 0, scale);
  int size = (dimX - scale) * (dimY - scale) * (dimZ - scale);
  int3* h_points2 = (int3 *) malloc(size / scale * sizeof(int3));
  int c = 0;
  for (int i = 0; i < size; ++i)
  {
    if(h_points[i].x >= 0)
    {
      memcpy(&h_points2[c++], &h_points[i], sizeof(int3));
    }
  }

  h_points2 = (int3 *) realloc(h_points2, c * sizeof(int3));
  h_points = (int3 *) realloc(h_points, c * sizeof(int3));
  memcpy(&h_points[0], &h_points2[0], c * sizeof(int3));
  free(h_points2);
  return c;

}
