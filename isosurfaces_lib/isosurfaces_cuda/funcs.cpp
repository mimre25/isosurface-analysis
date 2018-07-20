//
// Created by mimre on 8/3/16.
//
#include "funcs.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>
#include <visualization/hdr/HistogramVisualizer.h>
#include <filehandler/hdr/FloatBinaryWriter.h>
#include "isosurfaces_cuda/hdr/funcs.h"
#include "types/hdr/DistanceField.h"
#include "types/hdr/JointHistogram.h"
#include "isosurfaces_cuda/hdr/common.h"
#include "isosurfaces_cuda/hdr/helper_cuda.h"
#include "types/hdr/SimilarityMap.h"
#include "runtime/hdr/globals.h"

///DEBUG FUNCTIONS

int fixId(int idx)
{
  bool internal = idx >= 0;
  return internal ? idx : -idx -1;
}

void printBoundingBox(BoundingBox b)
{
  printf("min: %f, %f, %f ~ max: %f, %f, %f\n", b.minCorner.x, b.minCorner.y, b.minCorner.z, b.maxCorner.x, b.maxCorner.y, b.maxCorner.z);
}

void traversePrint(BVH bvh, string indent, int current)
{
  if (current >= 0)
  {
    InternalNode node = bvh.nodes[current];

    printf("Node id %d, l %d, r%d\n", node.id, node.leftChildId, node.rightChildId);
    printBoundingBox(node.boundingBox);

    traversePrint(bvh, indent + "L", node.leftChildId);
    traversePrint(bvh, indent + "R", node.rightChildId);
  } else
  {
    LeafNode leaf = bvh.leaves[fixId(current)];
    printf("Leaf id %d\n", leaf.id);
    printBoundingBox(leaf.boundingBox);
  }
}

void printHist(int* hist, int histSize)
{
  for (int i = 0; i < histSize; ++i)
  {
    printf("{");
    for (int j = 0; j < histSize; ++j)
    {
      printf("%d,", hist[i*histSize+j]);
    }
    printf("}\n");
  }
  printf("\n");
}

///END DEBUG FUNCTIONS




void calculateDistances_W(vector<float3> points, int dimX, int dimY, int dimZ, int dfDownscale, float* distances)
{
  vector<float3> pts;
  pts.reserve(points.size());
  for (float3 &p : points)
  {
    pts.push_back(make_float3(p.x,p.y,p.z));
  }
  calculate_distances_h(pts, dimX, dimY, dimZ, dfDownscale, distances);
}


long approximate_isosurfaces_W(float* data, float isovalue, const int dimX, const int dimY, const int dimZ, const int scale, int3** h_points)
{
  return approximate_isosurfaces_h(data, isovalue, dimX, dimY, dimZ, scale, h_points);
//  return approximate_isosurfaces_CPU_h(data, isovalue, dimX, dimY, dimZ, scale, h_points);
}


SimilarityMap calculate_histogram_W(vector<DistanceField> distanceFields, const int fieldSize_, const int histogramSize,
                                    const int similiarityMapSize, const bool multi)
{
  printf("Wrapper function start\n");
  unsigned long fieldSize = (unsigned long) fieldSize_;
  unsigned long numDistanceFields = distanceFields.size();

  float** dfields = new float*[numDistanceFields];
  float* minValues = new float[numDistanceFields];
  float* maxValues = new float[numDistanceFields];


  for(int i = 0; i < numDistanceFields; ++i)
  {
    unsigned long idx = static_cast<unsigned long>(i * fieldSize);
    dfields[i] = distanceFields[i].getDistancesAsFloatPointer();
    pair<float, float> interval = distanceFields[i].getInterval();
    minValues[i] = interval.first;
    maxValues[i] = interval.second;
  }

  cout << similiarityMapSize << endl;
  float* similarityMap = (float *) malloc(similiarityMapSize * similiarityMapSize * sizeof(*similarityMap));
  memset(similarityMap, 0, similiarityMapSize * similiarityMapSize * sizeof(*similarityMap));


  numDistanceFields = multi ? numDistanceFields/2 : numDistanceFields;

  size_t jointHistogramBytes = numDistanceFields * numDistanceFields * histogramSize * histogramSize * sizeof(int);
  size_t colRowBytes = numDistanceFields * numDistanceFields * histogramSize * sizeof(int);


  unsigned int* jointHist = (unsigned int *) malloc(jointHistogramBytes);
  unsigned int* colSums =   (unsigned int *) malloc(colRowBytes);
  unsigned int* rowSums =   (unsigned int *) malloc(colRowBytes);

  printf("calling host function\n");
  fflush(stdout);
  calculate_histogram_h(dfields, minValues, maxValues, fieldSize,
                        numDistanceFields, histogramSize, &jointHist[0], colSums, rowSums, similarityMap, similiarityMapSize, multi);




  SimilarityMap map = SimilarityMap(similarityMap, similiarityMapSize);


  free(similarityMap);
  free(jointHist);
  free(colSums);
  free(rowSums);
  delete[] dfields;
  delete[] minValues;
  delete[] maxValues;
  printf("Wrapper function done\n");

  return map;
}



void generateHierarchy_W(vector<float3> points, const int dimX, const int dimY, const int dimZ, float *distances, int numSamples, const int dfDownscale) {
  vector<float3> ps = vector<float3>();
  const int dimMax = max(dimX,max(dimY, dimZ));
  for (unsigned int i = 0; i < points.size() ; ++i)
  {
    float3 p = points[i];
    float3 pp = make_float3(p.x/dimMax, p.y/dimMax, p.z/dimMax);
    ps.push_back(pp);
  }
  generateHierarchy_h(ps, dimX, dimY, dimZ, distances, numSamples, dfDownscale, false, 0, ps);

};

void generateHierarchyi3_W(int3* points, const int dimX, const int dimY, const int dimZ, float *distances, const long numPoints, const int numSamples, const int dfDownscale) {
  vector<float3> ps = vector<float3>();
  const int dimMax = max(dimX,max(dimY, dimZ));
  for (int i = 0; i < numPoints ; ++i)
  {
    int3 p = points[i];
    float3 pp = make_float3(float(p.x)/dimMax, float(p.y)/dimMax, float(p.z)/dimMax);
    ps.push_back(pp);
  }
  generateHierarchy_h(ps, dimX, dimY, dimZ, distances, numSamples, dfDownscale, false, 0, ps);

};


void generateHierarchySurfaceToSurface_W(int3* points, const int dimX, const int dimY, const int dimZ, float *distances, const long numPoints, const int numSamples, const int dfDownscale, vector<float3> queryPoints, const long numQueryPoints) {
  vector<float3> ps = vector<float3>();

  const int dimMax = max(dimX,max(dimY, dimZ));

  for (int i = 0; i < numPoints; ++i)
  {
    int3 p = points[i];
    float3 pp = make_float3(float(p.x)/dimMax, float(p.y)/dimMax, float(p.z)/dimMax);
    ps.push_back(pp);
  }


  generateHierarchy_h(ps, dimX, dimY, dimZ, distances, numSamples, dfDownscale, true, numQueryPoints, queryPoints);

};

long findExtrema_W(float *data, int3 dimensions, float **h_points)
{
  return findExtrema_h(data, dimensions, h_points);
}
