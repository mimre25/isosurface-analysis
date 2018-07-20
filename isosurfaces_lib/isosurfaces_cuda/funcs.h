//
// Created by mimre on 8/3/16.
//
#ifndef __CUDA_WRAPPER__
#define __CUDA_WRAPPER__
#include <vector_types.h>
#include <vector_functions.h>
#include "isosurfaces_cuda/hdr/funcs.h"
#include "isosurfaces_cuda/hdr/BVH.h"
#include "types/hdr/DistanceField.h"
#include "types/hdr/JointHistogram.h"
#include "types/hdr/SimilarityMap.h"

extern "C"
void calculate_distances_h(vector<float3> points, int dimX, int dimY, int dimZ, int dfDownscale, float *h_distances);

void calculateDistances_W(vector<float3> points, int dimX, int dimY, int dimZ, int dfDownscale, float* distances);


extern "C"
long approximate_isosurfaces_CPU_h(float* h_data, float isovalue, const int dimX, const int dimY, const int dimZ, const int scale, int3 *h_points);

extern "C"
long approximate_isosurfaces_h(float* h_data, float isovalue, const int dimX, const int dimY, const int dimZ, const int scale, int3 **h_points);

long approximate_isosurfaces_W(float* data, float isovalue, const int dimX, const int dimY, const int dimZ, const int scale, int3 **h_points);


//extern "C"
//void calculate_histogram_h(float *distanceFields, float *minValues, float *maxValues, const int fieldSize,
//                           const int numFields, const int histogramSize, float* h_similarityMap, const int similarityMapSize);
extern "C"
void calculate_histogram_h(float **distanceFields, float *minValues, float *maxValues, const int fieldSize,
                           const unsigned long numFields, const unsigned long histogramSize,
                           unsigned int *h_jointHist, unsigned int *h_colSums, unsigned int *h_rowSums,
                           float *h_simMap, const int simMapSize, const bool multi);


SimilarityMap calculate_histogram_W(vector<DistanceField> distanceFields, const int fieldSize_, const int histogramSize,
                                    const int similiarityMapSize, const bool multi);


extern "C"
void generateHierarchy_h(vector<float3> points, const int dimX, const int dimY, const int dimZ, float *h_distances, const int numSamples, const int dfDownscale, const bool sampleGiven, const unsigned long sampleSize, vector<float3> samplePoints);

void generateHierarchy_W(vector<float3> points, const int dimX, const int dimY, const int dimZ, float* distances, int numSamples, const int dfDownscale);
void generateHierarchyi3_W(int3* points, const int dimX, const int dimY, const int dimZ, float* distances, const long numPoints, const int numSamples, const int dfDownscale);
void generateHierarchySurfaceToSurface_W(int3* points, const int dimX, const int dimY, const int dimZ, float *distances, const long numPoints, const int numSamples, const int dfDownscale, vector<float3> queryPoints, const long numQueryPoints);




extern "C"
long findExtrema_h(float *h_data, int3 dimensions, float **h_points);

long findExtrema_W(float *data, int3 dimensions, float **h_points);


#endif //__CUDA_WRAPPER__