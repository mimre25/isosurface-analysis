//
// Created by mimre on 7/27/16.
//

#ifndef ISOSURFACES_ISOSURFACE_H
#define ISOSURFACES_ISOSURFACE_H

#define POINT_CLOUDS false

#include <utility>
#include <string>
#include <vector>
#include <vector_types.h>


class Isosurface
{
public:
  void createPointCloud(int3 *points, const long len);
  void savePointCloudToFile(std::string filename);
  int* labelRegions(int3* points, const int3 dimensions, const long length);
  int removeSmallRegions(int3 *points, const int3 dimensions, const long length, const int threshold,
                           int3 **newPts);

  float* findExtremaValues();
  bool verifyJump(float isovalue);

  long calculateSurfacePoints(bool approx, const float curIsovalue, int3 **approxPoints);
  Isosurface(int dimX, int dimY, int dimZ, int dimT, float minV, float maxV);

  Isosurface();

  Isosurface(int dimX, int dimY, int dimZ, int dimT);

  int dimX;
  int dimY;
  int dimZ;
  int dimT;
  float minV;
  float maxV;
  float* img;
  float* extremaValues;
  long extremaValuesLength;
  std::vector< std::vector< std::vector<float> > > image;
  std::vector< std::pair< float3, float3 > > ptsAndNormals;
  std::vector<float3> points;
  std::vector< std::pair< float4, float3 > > ptsAndNormalsForPrint;
  std::vector<int> facesForPrint;
  std::vector< std::vector< std::vector<float> > > pointCloud;
  float* pointCloud2;


  void loadFile(std::string fileName, bool byte);

  std::vector<std::string> prepareStringVector();


  void parseDimension(std::string dim);

  void setMinMax(std::string minMax);

  float middleValue();

  float realMinV;
  float realMaxV;

  float* imageAsFloatPtr();

  void clear();

  bool isEmpty();

  long printBinary(unsigned char*& data);

  void freeImg();

private:
  bool imageCreated = false;
  void createImgPtr();

  void prePrint();

  bool extremaSet = false;


};


#endif //ISOSURFACES_ISOSURFACE_H
