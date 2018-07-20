//
// Created by mimre on 1/23/17.
//

#include "utils/hdr/floatNExtensions.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <math.h>
#include <cstdio>

__host__ __device__ bool operator==(const float4& left, const float4& right)
{
  return left.x == right.x && left.y == right.y && left.z == right.z && left.w == right.w;
}

bool operator!=(const float4& left, const float4& right)
{
  return !(left == right);
}

float3 operator+=(float3& left, const float3& right)
{
  left.x += right.x;
  left.y += right.y;
  left.z += right.z;
  return left;
}

float3 operator/=(float3& left, const float& r)
{
  left.x/=r;
  left.y/=r;
  left.z/=r;
  return left;
}

float3 operator/=(float3& left, const int& r)
{
  left.x/=(float)r;
  left.y/=(float)r;
  left.z/=(float)r;
  return left;
}

float3 normalize(const float3& v)
{
  float len = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
  float3 tmp = v;
  tmp /= len;
  return tmp;
}

bool operator<(const float4& left, const float4& right)
{
  if(left.x==right.x){
    if(left.y==right.y){
      if(left.z==right.z){
        return (left.w<right.w);
      }
      return (left.z<right.z);
    }
    return (left.y<right.y);
  }
  return (left.x<right.x);
}

void print(const float4& v)
{
  printf("%0.2f, %0.2f, %0.2f, %0.2f\n", v.x, v.y, v.z, v.w);
}
