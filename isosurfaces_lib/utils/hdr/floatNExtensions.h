//
// Created by mimre on 1/23/17.
//

#ifndef UTILS_FLOATNEXTENSIONS_H
#define UTILS_FLOATNEXTENSIONS_H


#include <vector_types.h>

__host__ __device__ bool operator==(const float4& left, const float4& right);

bool operator!=(const float4& left, const float4& right);
float3 operator+=(float3& left, const float3& right);

float3 operator/=(float3& left, const float& r);
float3 operator/=(float3& left, const int& r);
float3 normalize(const float3& v);
bool operator<(const float4& left, const float4& right);
void print(const float4& v);

#endif //UTILS_FLOATNEXTENSIONS_H
