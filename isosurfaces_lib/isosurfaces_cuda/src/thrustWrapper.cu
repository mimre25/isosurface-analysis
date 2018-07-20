#include "isosurfaces_cuda/hdr/thrustWrapper.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <utils/typeOperation.h>
#include "isosurfaces_cuda/hdr/common.h"


//THIS IS JUST A THRUST WRAPPER

using namespace thrust;


void sortByKeys(unsigned int* keys, float3* values, const int numValues)
{
  device_ptr<unsigned int> t_a(keys);
  device_ptr<float3> t_b(values);
  sort_by_key(t_a, t_a+numValues, t_b);
}

void findMinMax(float *d_points, float *min, float *max, const int size)
{



  thrust::device_ptr<float> t_pts(d_points);

// use max_element for reduction
  thrust::device_ptr<float> t_maxptr = thrust::max_element(t_pts, t_pts+size);
  thrust::device_ptr<float> t_minptr = thrust::min_element(t_pts, t_pts+size);

// retrieve result from device (if required)
  *max = t_maxptr[0];
  *min = t_minptr[0];


}


size_t removeDuplicates(vec4f* data, const size_t size, vec4f* outputVector)
{


  thrust::device_ptr<vec4f> t_vec(data);
  thrust::sort(t_vec, t_vec+size);
  thrust::device_ptr<vec4f> new_end = thrust::unique(thrust::device, t_vec, t_vec + size);
  thrust::device_ptr<vec4f> tmp(outputVector);
  thrust::device_ptr<vec4f> result = thrust::copy(t_vec, new_end, tmp);
  size_t length = thrust::distance(t_vec, new_end);
  printf("\nremove length = %li\n", length);
  return length;
}
