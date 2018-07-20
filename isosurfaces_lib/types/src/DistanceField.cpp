//
// Created by mimre on 8/1/16.
//

#include <iostream>
#include <limits>
#include <cmath>
#include <stdlib.h>
#include <set>
#include "types/hdr/DistanceField.h"
#include "filehandler/hdr/FloatBinaryWriter.h"
#include "isosurfaces_cuda/hdr/funcs.h"
#include "isosurfaces_cuda/funcs.h"
#include "utils/hdr/Report.h"
#include "runtime/hdr/globals.h"
#include "filehandler/hdr/BinaryFloatReader.h"
#include "isosurfaces_cuda/hdr/helpers.h"



const vector<float3> &DistanceField::getPoints() const
{
  return points;
}

void DistanceField::setPoints(const vector<float3> &points)
{
  DistanceField::points = points;
}

const vector<vector<vector<float>>> &DistanceField::getDistances()
{
  if(!distancesConverted)
  {
    distances.resize((unsigned long) dimX);
    for (int x = 0; x < dimX; ++x)
    {
      distances[x].resize((unsigned long) dimY);
      for (int y = 0; y < dimY; ++y)
      {
        distances[x][y].resize((unsigned long) dimZ);
        for (int z = 0; z < dimZ; ++z)
        {
          distances[x][y][z] = distancePtr[x * dimY * dimZ + y * dimZ + z];
        }
      }
    }
    distancesConverted = true;
  }

  return distances;
}

int DistanceField::getDimX() const
{
  return dimX;
}

void DistanceField::setDimX(int dimX)
{
  DistanceField::dimX = dimX;
}

int DistanceField::getDimY() const
{
  return dimY;
}

void DistanceField::setDimY(int dimY)
{
  DistanceField::dimY = dimY;
}

int DistanceField::getDimZ() const
{
  return dimZ;
}

void DistanceField::setDimZ(int dimZ)
{
  DistanceField::dimZ = dimZ;
}

float float3_distance(float3 v1, float3 v2)
{
  float x2 = (v2.x - v1.x) * (v2.x - v1.x);
  float y2 = (v2.y - v1.y) * (v2.y - v1.y);
  float z2 = (v2.z - v1.z) * (v2.z - v1.z);
  return sqrt(x2 + y2 + z2);
}

void DistanceField::calculateDistanceField(bool CPU, bool approx, int numSamples)
{
  if(CPU) {
    //this code is never used
    float distance;
    float cur;
    distances.resize((unsigned long) dimX);
    for (int x = 0; x < dimX; ++x) {
      distances[x].resize((unsigned long) dimY);
      for (int y = 0; y < dimY; ++y) {
        distances[x][y].resize((unsigned long) dimZ);
        for (int z = 0; z < dimZ; ++z) {
          cur = numeric_limits<float>::max();
          //do the calculations
          for (float3 &pt: points) {
            distance = float3_distance(make_float3(x*dfDownscale+dfDownscale/2, y*dfDownscale+dfDownscale/2, z*dfDownscale+dfDownscale/2), pt);
            if (distance < cur) {
              distances[x][y][z] = distance;
              cur = distance;
            }
          }
        }
        cout << x << ": " << y << "/" << dimY << endl;
        fflush(stdout);
      }
      cout << x << "/" << dimX << endl;
    }
  } else
  {

    this->distancePtr = (float *) malloc((size_t) (dimX * dimY * dimZ)*sizeof(float));
    if (approx)
    {
      vector<float3> vec;
      generateHierarchyi3_W(pointsi3, dimXOrig, dimYOrig, dimZOrig, distancePtr, numPoints, numSamples, dfDownscale);///BVH

    } else {
      generateHierarchy_W(points, dimXOrig, dimYOrig, dimZOrig, distancePtr, numSamples, dfDownscale);///BVH
    }


    distancePtrSet = true;


    points.clear();
    points.shrink_to_fit();
  }
}

DistanceField::DistanceField()
{}


void DistanceField::writeToFile(string fileName)
{
  filehandler::FloatBinaryWriter floatBinaryWrtier(fileName);
  floatBinaryWrtier.writeFile(fileName, distancePtr, dimX*dimY*dimZ);

}

void DistanceField::clear()
{
  distances.clear();
}



pair<float, float> DistanceField::getInterval()
{
  float min = numeric_limits<float>::max();
  float max = numeric_limits<float>::min();
  if (distancePtrSet)
  {
    for (int i = 0; i < dimX * dimY * dimZ; ++i)
    {
      float val = distancePtr[i];
      if (val < min)
      {
        min = val;
      }
      if (val > max)
      {
        max = val;
      }
    }
  }
  else {
    for (int x = 0; x < dimX; ++x)
    {
      for (int y = 0; y < dimY; ++y)
      {
        for (int z = 0; z < dimZ; ++z)
        {
          float val = distances[x][y][z];
          if (val < min)
          {
            min = val;
          }
          if (val > max)
          {
            max = val;
          }
        }
      }
    }
  }

  return pair<float, float>(min, max);
}

float *DistanceField::getDistancesAsFloatPointer()
{
  if (!distancePtrSet)
  {
    this->distancePtr = (float *) malloc(unsigned(dimX * dimY * dimZ * sizeof(float)));
    int i = 0;
    for (int x = 0; x < dimX; ++x)
    {
      for (int y = 0; y < dimY; ++y)
      {
        for (int z = 0; z < dimZ; ++z)
        {
          distancePtr[i++] = distances[x][y][z];
        }
      }
    }
  }
  distancePtrSet = true;
  return distancePtr;
}


DistanceField::DistanceField(const vector<float3> &points, int dfDownscale, int dimX, int dimY, int dimZ, int dimXOrig, int dimYOrig,
                             int dimZOrig) : points(points), dfDownscale(dfDownscale), dimX(dimX), dimY(dimY), dimZ(dimZ), dimXOrig(dimXOrig),
                                             dimYOrig(dimYOrig), dimZOrig(dimZOrig)
{
  pointsi3 = (int3 *) malloc(points.size() * sizeof(int3));
}

DistanceField::DistanceField(int3 *pointsi3, int dfDownscale, int dimX, int dimY, int dimZ, int dimXOrig, int dimYOrig, int dimZOrig, long numPoints)
    : pointsi3(pointsi3), dfDownscale(dfDownscale), dimX(dimX), dimY(dimY), dimZ(dimZ), dimXOrig(dimXOrig), dimYOrig(dimYOrig), dimZOrig(dimZOrig), numPoints(numPoints)
{}

void DistanceField::loadFromFile(string s, long size)
{
  filehandler::BinaryFloatReader binaryFloatReader = filehandler::BinaryFloatReader(s);
  distancePtr = (float *) malloc(size * sizeof(float));
  binaryFloatReader.readBytes(size, distancePtr);
  distancePtrSet = true;
}

DistanceField::DistanceField(int dfDownscale, int dimXOrig, int dimYOrig, int dimZOrig) : dfDownscale(dfDownscale), dimX(dimXOrig/dfDownscale),
                                                                         dimY(dimYOrig/dfDownscale), dimZ(dimZOrig/dfDownscale),
                                                                         dimXOrig(dimXOrig), dimYOrig(dimYOrig),
                                                                         dimZOrig(dimZOrig)
{}

void DistanceField::print()
{
  if(!distancePtrSet)
  {
    getDistancesAsFloatPointer();
  }
  for (int x = 0; x < dimX; ++x)
  {
    printf("{");
    for (int y = 0; y < dimY; ++y)
    {
      printf("[");
      for (int z = 0; z < dimZ; ++z)
      {
        float p = distancePtr[x*dimY*dimZ+y*dimZ+z];
        printf("%0.2f, ", p);
      }
      printf("]\n");
    }
    printf("}\n");
  }
}

float *DistanceField::getDistancePtr() const
{
  return distancePtr;
}

void DistanceField::setDistancePtr(float *distancePtr)
{
  DistanceField::distancePtr = distancePtr;
  distancePtrSet = true;
}

bool DistanceField::isDistancePtrSet() const
{
  return distancePtrSet;
}

void DistanceField::calculateSurfaceToSurfaceDistanceField(const int numPreSamples, int3 *queryPoints,
                                                            long lengthOfQueryPoints)
{
  this->distancePtr = (float *) malloc((size_t) (lengthOfQueryPoints)*sizeof(float));

  std::set<int3> blocks;
  const int dimMax = max(dimXOrig,max(dimYOrig, dimZOrig));

  for (int i = 0; i < lengthOfQueryPoints ; ++i)
  {
    int3 p = queryPoints[i];
    int3 block = make_int3(p.x / dfDownscale, p.y / dfDownscale, p.z / dfDownscale);
    blocks.insert(block);

  }
  vector<float3> ps = vector<float3>();
  for (std::set<int3>::iterator it=blocks.begin(); it!=blocks.end(); ++it)
  {
    int3 block = *it;
    float3 pp = make_float3(
        float(block.x*dfDownscale + dfDownscale/2)/dimMax,
        float(block.y*dfDownscale + dfDownscale/2)/dimMax,
        float(block.z * dfDownscale + dfDownscale/2)/dimMax);
    ps.push_back(pp);
  }
  this->numDistance = ps.size();
  generateHierarchySurfaceToSurface_W(pointsi3, dimXOrig, dimYOrig, dimZOrig, distancePtr, numPoints, numPreSamples, dfDownscale, ps, ps.size());///BVH

}

long DistanceField::getNumPoints() const
{
  return numPoints;
}

long DistanceField::getNumberOfSamples() const
{
  return dimX*dimY*dimZ;
}


void DistanceField::upscale()
{
  vector<vector<vector<float>>> newDistances = vector<vector<vector<float>>>();

  newDistances.resize((unsigned long) dimXOrig);
  for (int x = 0; x < dimXOrig; ++x)
  {
    newDistances[x].resize((unsigned long) dimYOrig);
    for (int y = 0; y < dimYOrig; ++y)
    {
      newDistances[x][y].resize((unsigned long) dimZOrig);
      for (int z = 0; z < dimZOrig; ++z)
      {
        if(x % dfDownscale == 0 && y % dfDownscale == 0 && z % dfDownscale == 0)
        {
          newDistances[x][y][z] = getDistances()[x/dfDownscale][y/dfDownscale][z/dfDownscale];
        } else
        {
          newDistances[x][y][z] = static_cast<float>(-1e5);
        }
      }
    }
  }
  distances = newDistances;
  distancePtrSet = false;
  dfDownscale = 1;
  dimX = dimXOrig;
  dimY = dimYOrig;
  dimZ = dimZOrig;
}

int DistanceField::getDfDownscale() const
{
  return dfDownscale;
}

int DistanceField::getDimXOrig() const
{
  return dimXOrig;
}

int DistanceField::getDimYOrig() const
{
  return dimYOrig;
}

int DistanceField::getDimZOrig() const
{
  return dimZOrig;
}

/**
 * Computes the difference between this DistanceField and the other
 * @param other DistanceField to compare to.
 * @return New DistanceField object with the difference in as distances
 */
DistanceField DistanceField::difference(DistanceField &other)
{
  vector<vector<vector<float>>>  dis = this->getDistances();
  vector<vector<vector<float>>>  otherDis = other.getDistances();
  vector<vector<vector<float>>> diff(dis.size());
  float max = -1;
  for (int i = 0; i < dis.size(); ++i)
  {
    diff[i].resize(dis[i].size());
    for (int j = 0; j < dis[i].size(); ++j)
    {
      diff[i][j].resize(dis[i][j].size());
      for (int k = 0; k < dis[i][j].size(); ++k)
      {
        diff[i][j][k] = abs(dis[i][j][k] - otherDis[i][j][k]);
        if(diff[i][j][k] > max)
        {
          max = diff[i][j][k];
        }
      }
    }
  }
  // normalizing
  for (int i = 0; i < dis.size(); ++i)
  {
    for (int j = 0; j < dis[i].size(); ++j)
    {
      for (int k = 0; k < dis[i][j].size(); ++k)
      {
        diff[i][j][k] = diff[i][j][k]/max;
      }
    }
  }
  DistanceField diffField = DistanceField();
  diffField.dimX = this->dimX;
  diffField.dimY = this->dimY;
  diffField.dimZ = this->dimZ;
  diffField.dimXOrig = this->dimXOrig;
  diffField.dimYOrig = this->dimYOrig;
  diffField.dimZOrig = this->dimZOrig;
  diffField.distances = diff;
  diffField.distancePtrSet = false;
  return diffField;
}

/**
 * saves the distanceField to file in binary representation
 * @param filename the target file name
 */
void DistanceField::saveToFile(string filename)
{

  filehandler::FloatBinaryWriter floatBinaryWriter(filename);
  floatBinaryWriter.writeFile(filename, this->getDistancesAsFloatPointer(), this->getNumberOfSamples());

}

long DistanceField::getNumDistance() const
{
  return numDistance;
}

void DistanceField::setNumDistance(long numDistance)
{
  DistanceField::numDistance = numDistance;
}




