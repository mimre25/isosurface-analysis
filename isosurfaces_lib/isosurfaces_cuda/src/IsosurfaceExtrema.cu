#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "isosurfaces_cuda/hdr/common.h"
#include "isosurfaces_cuda/hdr/thrustWrapper.h"
#include "isosurfaces_cuda/hdr/helpers.h"




#define NEIGHBORHOOD_SIZE 5
#define THRESHOLD 5


using namespace std;




__host__ __device__ float checkNeighborhood(int3 pos, float* data, int3 dimensions)
{
  int dimX = dimensions.x;
  int dimY = dimensions.y;
  int dimZ = dimensions.z;

  int x = pos.x;
  int y = pos.y;
  int z = pos.z;
  float pointVal = data[x+y*dimY + z*dimY*dimZ];

  int numHigher = 0;
  int numLower = 0;
  int numSame = 0;
  for (int i = -1; i < -1+NEIGHBORHOOD_SIZE; ++i)
  {
    int newX = x + i;
    for (int j = -1; j < -1+NEIGHBORHOOD_SIZE; ++j)
    {
      int newY = y + j;
      for (int k = -1; k < -1+NEIGHBORHOOD_SIZE; ++k)
      {
        int newZ = z + k;
        //boundary check
        if (newX > -1 && newY > -1 && newZ > -1
            && newX < dimX && newY < dimY && newZ < dimZ)
        {
          float newVal = data[newX + newY * dimY + newZ * dimY * dimZ];
          numHigher = newVal > pointVal ? numHigher + 1 : numHigher;
          numLower = newVal < pointVal ? numLower + 1 : numLower;
          numSame = newVal == pointVal ? numSame + 1 : numSame;
        }
      }
    }
  }

  //check if the value passes the threshold for equal values, or if it's not an extrema
  float outVal = pointVal;
  if (numSame > THRESHOLD)
  {
    outVal = static_cast<float>(DUMMY_VALUE);
  }
  if (numLower > 0 && numHigher > 0)
  {
    outVal = static_cast<float>(DUMMY_VALUE);
  }

  return outVal;
}



__global__ void find_extrema_g(float* data, int3 dimensions, const int size, const int offset, float* extrema)
{
  int idx = threadIdx.x + (blockIdx.x+offset) * blockDim.x;
  if (idx < size)
  {
    int3 point = findPointForIndex(idx, dimensions);
    extrema[findOutputIndex(point, dimensions)] = checkNeighborhood(point, data, dimensions);
  }
}


void find_extrema_CPU(float* data, int3 dimensions, const int size, const int offset, float* extrema)
{
  for (int idx = 0; idx < size; ++idx)
  {
    if (idx < size)
    {
      int3 point = findPointForIndex(idx, dimensions);
      extrema[findOutputIndex(point, dimensions)] = checkNeighborhood(point, data, dimensions);
    }
  }

}




extern "C"
__host__ long findExtrema_h(float *h_data, const int3 dimensions, float **h_points)
{
  //set up
  unsigned int NUM_THREADS = 256;
  unsigned int MAX_NUM_BLOCKS = (2<<14);
  unsigned long NUM_POINTS = (unsigned long) (dimensions.x * dimensions.y * dimensions.z);
  unsigned long POINTS_SIZE = NUM_POINTS * sizeof(float);
  unsigned long OUT_SIZE = NUM_POINTS * sizeof(float);

  float *d_data;
  float *d_points;
  float *d_points2;

  cudaMalloc((void **) &d_points, OUT_SIZE);
  cudaMemset(d_points, 0xff, OUT_SIZE);
  cudaMalloc((void **) &d_data, POINTS_SIZE);

  cudaMemcpy(d_data, h_data, POINTS_SIZE, cudaMemcpyHostToDevice);


  //run

  int blockNumber = calculateBlockNumber(NUM_POINTS, NUM_THREADS);

  for (int offset = 0; offset < blockNumber; offset += MAX_NUM_BLOCKS)
  {
    find_extrema_g <<< MAX_NUM_BLOCKS, NUM_THREADS >>>(d_data, dimensions, NUM_POINTS, offset, d_points);
  }




  //compact
  cudaMalloc((void **) &d_points2, OUT_SIZE);

  cudaMemset(d_points2, 0, OUT_SIZE);

  long length = compact(d_points, d_points2, NUM_POINTS, is_not_dummy_value());


  *h_points = (float*) malloc(length * sizeof(float));

  //cleanup
  cudaMemcpy(*h_points, d_points2, length*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_points2);
  cudaFree(d_data);
  cudaFree(d_points);

  return length;

}

