//
// Created by mimre on 8/3/16.
//
#ifndef __HELPERS__
#define __HELPERS__

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.h>




inline __host__ __device__ bool operator==(float3 a, float3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __host__ __device__ bool operator==(int3 a, int3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __host__ __device__ bool operator<(const int3 a, const int3 b)
{
  if (a.x < b.x)
  {
    return true;
  } else if (a.y < b.y)
  {
    return true;
  } else
  {
    return a.z < b.z;
  }
}

inline __host__ __device__ int3 operator+(const int3 & a, const int3 & b) {

  return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);

}

inline __host__ __device__ int findOutputIndex(int3 pt, const int dimX, const int dimY, const int dimZ, const int scale)
{
  return  (pt.x * ((dimY-scale) * (dimZ-scale)) + pt.y * (dimZ -scale) + pt.z)/scale;
}



inline __host__ __device__ int findOutputIndex(int3 pt, int3 dimensions)
{
  int dimY = dimensions.y;
  int dimZ = dimensions.z;
  return  pt.x * dimY * dimZ + pt.y * dimZ + pt.z;
}

//gives back the index for a point
inline __host__ __device__ int findPointIndex(int3 pt, const int dimX, const int dimY, const int dimZ)
{
  return pt.x * (dimY * dimZ) + pt.y * dimZ + pt.z;
}

//finding the sample point in the grid
//eg (0,0,0) is the first in total
inline __host__ __device__ int3 findPointForIndex(int idx, int3 dimensions)
{
  int dimZ = dimensions.z;
  int dimY = dimensions.y;
  int z = idx % dimZ;
  int y = (idx / dimZ) % dimY;
  int x = idx / (dimY * dimZ);

  return make_int3(x, y, z);

}

inline __host__ __device__ int3 findPointForIndex(int idx, int dimY, int dimZ)
{
  int z = idx % dimZ;
  int y = (idx / dimZ) % dimY;
  int x = idx / (dimY * dimZ);

  return make_int3(x, y, z);

}
#endif //__HELPERS__