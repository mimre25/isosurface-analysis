//
// Created by mimre on 8/1/16.
//

#ifndef ISOSURFACES_DISTANCEFIELD_H
#define ISOSURFACES_DISTANCEFIELD_H


#include <vector>
#include <string>
#include <boost/math/tools/tuple.hpp>
#include <vector_types.h>

using namespace std;

class DistanceField
{
public:

  void saveToFile(string filename);

  DistanceField difference(DistanceField& other);

  int getDfDownscale() const;

  int getDimXOrig() const;

  int getDimYOrig() const;

  int getDimZOrig() const;

  DistanceField(const vector<float3> &points, int dfDownscale, int dimX, int dimY, int dimZ, int dimXOrig, int dimYOrig, int dimZOrig);
  DistanceField(int3 *pointsi3, int dfDownscale, int dimX, int dimY, int dimZ, int dimXOrig, int dimYOrig, int dimZOrig, long numPoints);

  void loadFromFile(string s, long size);
  void calculateDistanceField(bool CPU, bool approx, int numSamples);

  void calculateSurfaceToSurfaceDistanceField(const int numPreSamples, int3 *queryPoints, long lengthOfQueryPoints);

  const vector<float3> &getPoints() const;

  void setPoints(const vector<float3> &points);

  const vector< vector< vector< float> > > &getDistances();
  float* getDistancesAsFloatPointer();

  int getDimX() const;



  void setDimX(int dimX);

  int getDimY() const;

  DistanceField(int dfDownscale, int dimXOrig, int dimYOrig, int dimZOrig);

  DistanceField();

  void setDimY(int dimY);

  int getDimZ() const;

  void setDimZ(int dimZ);


  void writeToFile(std::string fileName);

  pair<float,float> getInterval();
  void print();

  void upscale();

private:
  vector<float3> points;
  int3* pointsi3;

  vector< vector< vector<float> > > distances;
  int dfDownscale;
  int dimX;
  int dimY;
  int dimZ;
  int dimXOrig;
  int dimYOrig;
  int dimZOrig;
  float* distancePtr;
  bool distancePtrSet = false;

  //stores how many points the isosurface contains
  long numPoints;

  //stores how many distance sample points this distancefield has
  long numDistance;

public:
  long getNumberOfSamples() const;

  long getNumDistance() const;

  void setNumDistance(long numDistance);

public:
  long getNumPoints() const;

private:
  void clear();

  bool distancesConverted = false;
public:
  float *getDistancePtr() const;

  bool isDistancePtrSet() const;

  void setDistancePtr(float *distancePtr);


  int id;

};


#endif //ISOSURFACES_DISTANCEFIELD_H
