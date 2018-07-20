#ifndef _MARCHING_CUBE_H
#define _MARCHING_CUBE_H

#include "utils/typeOperation.h"

typedef struct {
	vec3f v;
	int vid;
} MCSortVertex;

void marching_cube(const float& iso_val, const vec3i& dim, float*** data, std::vector<MCSortVertex>& ret_points);
void cudaMarchingCube(const float& iso_val, const vec3i& dim, float* data, const int& maxVerts, std::vector<vec4f>& ret_points);

void remove_duplicate(std::vector<vec4f>& ret, std::vector<int>& indices, std::vector<MCSortVertex>& points);
void remove_duplicate(std::vector<vec4f>& points);

#endif