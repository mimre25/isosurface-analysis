//
// Created by mimre on 8/3/16.
//
#ifndef __THRUST_WRAPPER__
#define __THRUST_WRAPPER__


#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "isosurfaces_cuda/hdr/common.h"
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <stdlib.h>
#include "utils/typeOperation.h"


struct is_positive
{
  __host__ __device__
  bool operator()(const int3 p)
  {
    return (p.x >= 0);
  }
};

#define DUMMY_VALUE -1e10
#define EPSILON 2^-20

struct is_not_dummy_value
{
  __host__ __device__ bool operator()
      (const float &a)
  {
    float diff = abs(a - DUMMY_VALUE);
    return a > DUMMY_VALUE;

  }
};

void sortByKeys(unsigned int* keys, float3* values, const int numValues);

void findMinMax(float *d_points, float *min, float *max, const int size);

template<typename type, typename Predicate> long compact(type* d_inputArray, type* d_outputArray, const int size, Predicate predicate)
{
  thrust::device_ptr<type> t_inputArray(d_inputArray);
  thrust::device_ptr<type> t_outputArray(d_outputArray);
  thrust::device_ptr<type> result = thrust::copy_if(t_inputArray, t_inputArray + size, t_outputArray, predicate);
  long length = thrust::distance(t_outputArray, result);
  return length;
}

/**
 *
 * @param data on gpu
 * @param size
 * @param outPutVector pointer to array/vector where result should be stored in
 */
size_t removeDuplicates(vec4f* data, const size_t size, vec4f* outPutVector);


#endif //__THRUST_WRAPPER__