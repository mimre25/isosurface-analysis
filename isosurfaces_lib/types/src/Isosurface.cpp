//
// Created by mimre on 7/27/16.
//

#include <sstream>
#include <iomanip>
#include <cstring>
#include <limits>
#include <iostream>
#include <utils/hdr/Report.h>
#include <runtime/hdr/DistanceEqualization.h>
#include <filehandler/hdr/FloatBinaryWriter.h>
#include <isosurfaces_cuda/hdr/MarchingCube.h>
#include <set>
#include "isosurfaces_cuda/hdr/helpers.h"
#include "types/hdr/Isosurface.h"
#include "filehandler/hdr/FileReader.h"
#include "utils/hdr/StringUtils.h"
#include "runtime/hdr/globals.h"
#include "filehandler/hdr/BinaryFloatReader.h"
#include "isosurfaces_cuda/funcs.h"
#include "utils/hdr/MemUtils.h"
#include "utils/hdr/floatNExtensions.h"
#include "utils/typeOperation.h"


using namespace filehandler;

Isosurface::Isosurface()
{}

void Isosurface::loadFile(string fileName, bool byte)
{

  string datafile = fileName + DATA_FILE_EXTENSION;
cout << datafile << endl;

  BinaryFloatReader binaryFloatReader(datafile);
  if (byte)
  {
    binaryFloatReader.readBytes(dimX, dimY, dimZ, image);
    dimX /= INPUT_DOWNSCALE;
    dimY /= INPUT_DOWNSCALE;
    dimZ /= INPUT_DOWNSCALE;
  } else {
    binaryFloatReader.read(dimX, dimY, dimZ, image);
  }
  float max = -100000;
  float min = 100000;
  for (int x = 0; x < dimX; ++x)
  {
    for (int y = 0; y < dimY; ++y)
    {
      for (int z = 0; z < dimZ; ++z)
      {
        float val = image[x][y][z];
        if (val > max)
        {
          if(val <= this->maxV)
          {
            max = val;
          }
        }
        if (val < min)
        {
          if (val >= this->minV)
          {
          min = val;
          }
        }
      }
    }
  }
  realMinV = min;
  realMaxV = max;
}


vector<string> Isosurface::prepareStringVector()
{
  vector<string> result;
  vector<string> vertices;
  vector<string> normals;
  stringstream pt;
  stringstream n;
  for (pair<float4, float3> p : ptsAndNormalsForPrint)
  {

    pt << "v " << p.first.x << " " << p.first.y << " " << p.first.z << endl;


    n << "vn " << p.second.x << " " << p.second.y << " " << p.second.z << endl;
  }
  result.push_back(pt.str());
  result.push_back(n.str());
  unsigned long max = ptsAndNormals.size();
  result.push_back(string());
  result.push_back(string());
  stringstream f;
  for (unsigned long j = 0; j < max; j += 3)
  {
      int i = facesForPrint[j]+1;
      int k = facesForPrint[j+1]+1;
      int m = facesForPrint[j+2]+1;
      f << "f " << i << "//" << i
        << " " << k << "//" << k
        << " " << m << "//" << m << endl;
  }

  result.push_back(f.str());

  return result;
}

void Isosurface::parseDimension(string dim)
{
  vector<string> dimensions = utils::StringUtils::split(dim, ' ');
  unsigned long len = dimensions.size();

  switch (len) {
    case 4:
      dimT = stoi(dimensions[3]);
    case 3:
      dimZ = stoi(dimensions[2]);
    case 2:
      dimY = stoi(dimensions[1]);
    case 1:
      dimX = stoi(dimensions[0]);
      break;
    default:
      throw("Dimension mismatch: " + dim);

  }
}

void Isosurface::setMinMax(string minMax)
{
  vector<string> values = utils::StringUtils::split(minMax, ' ');
  unsigned long len = values.size();
  switch (len) {
    case 2:
      maxV = stof(values[1]);
    case 1:
      minV = stof(values[0]);
      break;
    default:
      throw("Min/Max values count mismatch " + minMax);
  }
}

Isosurface::Isosurface(int dimX, int dimY, int dimZ, int dimT, float minV, float maxV) : dimX(dimX), dimY(dimY),
                                                                                         dimZ(dimZ), dimT(dimT),
                                                                                         minV(minV), maxV(maxV)
{
}

Isosurface::Isosurface(int dimX, int dimY, int dimZ, int dimT) : dimX(dimX), dimY(dimY), dimZ(dimZ), dimT(dimT)
{}

float Isosurface::middleValue()
{
  return (realMinV + realMaxV) / 2;
}

void Isosurface::clear()
{
  ptsAndNormals.clear();
  points.clear();
  if(POINT_CLOUDS)
  {
    pointCloud.clear();
    free(pointCloud2);
  }
}

float *Isosurface::imageAsFloatPtr()
{
  if (!imageCreated) {
    createImgPtr();
  }
  return img;

}

void Isosurface::createImgPtr()
{
  this->img = (float *) malloc(dimX * dimY * dimZ * sizeof(float));
  int i = 0;
  for (int x = 0; x < dimX; ++x)
  {
    for (int y = 0; y < dimY; ++y)
    {
      for (int z = 0; z < dimZ; ++z)
      {
        img[i++] = image[x][y][z];
      }
    }
  }
  imageCreated = true;
}

/**
 * creates a pointcloud from the approximation array as a 3d vector
 * @param points the approximation points
 * @param len number of points int he approximation
 */
void Isosurface::createPointCloud(int3 *points, const long len)
{
  if(POINT_CLOUDS)
  {
    pointCloud2 = (float *) (malloc(dimX * dimY * dimZ * sizeof(*pointCloud2)));
    memset(pointCloud2, 0, dimX * dimY * dimZ * sizeof(*pointCloud2));
    pointCloud.resize(static_cast<unsigned long>(dimX));
    for (int i = 0; i < dimX; ++i)
    {
      pointCloud[i].resize(static_cast<unsigned long>(dimY));
      for (int j = 0; j < dimY; ++j)
      {
        pointCloud[i][j].resize(static_cast<unsigned long>(dimZ));
        for (int k = 0; k < dimZ; ++k)
        {
          pointCloud[i][j][k] = 0.0f;

        }
      }
    }
    for (int i = 0; i < len; ++i)
    {
      int3 p = points[i];
      pointCloud[p.x][p.y][p.z] = 1.0f;
      pointCloud2[p.x * dimY * dimZ + p.y * dimZ + p.z] = 1.0f;
    }
  }
}

/**
 * stores the @member pointcloud to the given filename
 * @param filename the name of the file to be written to
 */
void Isosurface::savePointCloudToFile(string filename)
{
  if (POINT_CLOUDS)
  {
    filehandler::FloatBinaryWriter floatBinaryWriter(filename);

    floatBinaryWriter.writeFile(filename, pointCloud2, dimX * dimY * dimZ);
  }
}

long Isosurface::calculateSurfacePoints(bool approx, const float curIsovalue, int3 **approxPoints)
{
  Report::begin(DistanceEqualization::currentStage + DistanceEqualization::APPROX);
  this->points.clear();
  vector<float3> currentPoints;
  MarchingCubes marchingCubes(this);

  long length;
  if (!approx)
  {
    std::vector<vec4f> vertices_gpu;
    vec3i dim = makeVec3i(dimX, dimY, dimZ);
    Report::begin("marching cubes");
    cudaMarchingCube(curIsovalue, dim, this->imageAsFloatPtr(), dimX*dimY*dimZ, vertices_gpu);
    Report::end("marching cubes");

    printf("size afterwards: %lu\n", vertices_gpu.size());


    Report::begin("transform");
    std::transform(vertices_gpu.cbegin(), vertices_gpu.cend(), std::back_inserter(this->points),
                   [](const vec4f& v) {
      float3 f;
      f.x = v.x;
      f.y = v.y;
      f.z = v.z;
      return f;
    });
    Report::end("transform");


    length = this->points.size();

  } else
  {
    int scale = 1;

    length = approximate_isosurfaces_W(this->imageAsFloatPtr(), curIsovalue, dimX, dimY, dimZ, scale, approxPoints);
  }
  Report::end(DistanceEqualization::currentStage + DistanceEqualization::APPROX);
  return length;
}

bool Isosurface::isEmpty()
{
  return points.empty();
}

long Isosurface::printBinary(unsigned char*& data)
{
  prePrint();
  unsigned int n = ptsAndNormalsForPrint.size();
  unsigned int m = facesForPrint.size();
  data = new unsigned char[n*(sizeof(float4)+sizeof(float3))+(m+2)*sizeof(int)];
  float4* pts = (float4*)&data[sizeof(int)];
  float3* normals = (float3*)&data[sizeof(int)+n*sizeof(float4)];
  int* faces = (int*)&data[2*sizeof(int)+n*(sizeof(float4)+sizeof(float3))];
  int* num_pts = (int*)&data[0];
  int* num_faces = (int*)&data[sizeof(int)+n*(sizeof(float4)+sizeof(float3))];

  num_pts[0] = n;
  num_faces[0] = m;
  for (unsigned int j = 0; j < n; ++j)
  {
    pts[j] = ptsAndNormalsForPrint[j].first;
    normals[j] = ptsAndNormalsForPrint[j].second;
  }
  for (unsigned int i = 0; i < m; ++i)
  {
    faces[i] = facesForPrint[i];
  }
  
  return n*(sizeof(float4)+sizeof(float3))+(m+2)*sizeof(int);
}



struct pointInfo
{
public:
  float4 v;
  float3 normal;
  int vidx;

  bool operator< (const pointInfo& other)
  {
    bool result = (v < other.v);
    return result;
  }

  pointInfo& operator= (const pointInfo& pInfo)
  {
    this->v = pInfo.v;
    this->normal = pInfo.normal;
    this->vidx = pInfo.vidx;
    return *this;
  }
  void printInfo()
  {
    printf("%d, %0.2f, %0.2f\n", this->vidx, this->v.x, this->normal.x);
  }
};




void Isosurface::prePrint()
{
  vector<struct pointInfo> pointInfos(ptsAndNormals.size());
  facesForPrint = vector<int>(ptsAndNormals.size());
  for (int i = 0; i < (int) ptsAndNormals.size(); ++i)
  {
    pointInfos[i] = {
        make_float4(ptsAndNormals[i].first.x, ptsAndNormals[i].first.y, ptsAndNormals[i].first.z,
                    1.0),//where is points defined?
        ptsAndNormals[i].second,
        i
    };
  }


  sort(pointInfos.begin(), pointInfos.end());

  float4 *tmp_vertices = new float4[pointInfos.size()];
  float3 *tmp_normals = new float3[pointInfos.size()];

  int num_unique_points = 0, vid;
  int num_normal = 0;
  float4 prev = make_float4(-1e30, -1e30, -1e30, -1e30), curr;
  float3 normal;
  for (unsigned int i = 0; i < pointInfos.size(); ++i)
  {
    curr = pointInfos[i].v;
    normal = pointInfos[i].normal;
    vid = pointInfos[i].vidx;

    if (curr != prev)
    {
      prev = curr;
      tmp_vertices[num_unique_points] = make_float4(curr.x, curr.y, curr.z, 1.0f);
      tmp_normals[num_unique_points] = make_float3(0, 0, 0);
      if (num_normal != 0)
      {
        tmp_normals[num_unique_points -
                    1] /= num_normal; //we need to recalculate normal since the same vertex may have different normals on different faces
        tmp_normals[num_unique_points - 1] = normalize(tmp_normals[num_unique_points - 1]);
        num_normal = 0;
      }
      ++num_unique_points;
    }
    ++num_normal;
    tmp_normals[num_unique_points - 1] += normal;
    facesForPrint[vid] = num_unique_points - 1;
  }
  if (num_normal != 0)
  {
    tmp_normals[num_unique_points -
                1] /= num_normal; //we need to recalculate normal since the same vertex may have different normals on different faces
    tmp_normals[num_unique_points - 1] = normalize(tmp_normals[num_unique_points - 1]);
  }

  ptsAndNormalsForPrint = vector<pair<float4, float3>>((unsigned long) num_unique_points);
  for (int i = 0; i < num_unique_points; ++i)
  {
    ptsAndNormalsForPrint[i] = make_pair(tmp_vertices[i], tmp_normals[i]);
  }

  delete[] tmp_vertices;
  delete[] tmp_normals;
}

void Isosurface::freeImg()
{
  if(imageCreated)
  {
    free(this->img);
    imageCreated = false;
  }
  if(extremaSet)
  {
    free(this->extremaValues);
    extremaSet = false;
  }

}

float *Isosurface::findExtremaValues()
{




  extremaValuesLength = findExtrema_W(this->imageAsFloatPtr(), make_int3(dimX, dimY, dimZ), &extremaValues);

  extremaSet = true;

  cout << "num extremaValues " <<  extremaValuesLength << endl;
  cout << "exit";

}

/**
 * Verifies if the given isovalue is an actual jump
 *
 * Current implementation uses the extremaValues around the isovalue
 *
 * @param isovalue the value in question
 * @return true if it's a real jump, false otherwise
 */
bool Isosurface::verifyJump(float isovalue)
{
  float threshold = 5.0f;
  for (int i = 0; i < extremaValuesLength; ++i)
  {
    if(abs(extremaValues[i]-isovalue) < threshold)
    {
      return true;
    }
  }
  return false;
}


/**
 * Labels regions in the surface given by the points array.
 * The points array is an array of positions of existing voxels
 * All points p hold (0,0,0) <= p < dimensions
 * Labels the voxel accordingly and returns an integer array with the labels for each point
 * @param points voxel positions
 * @param dimensions volume dimensions
 * @param length number of points
 * @return array with length labels, one for each voxel
 */
int* Isosurface::labelRegions(int3* points, const int3 dimensions, const long length)
{
  int label = 1;
  int* labels = (int*)(malloc(length * sizeof(*labels)));
  memset(labels, 0, length * sizeof(*labels));
  std::map<int, int> equalLabels;


  for (int i = 0; i < length; ++i)
  {
    int3 p = points[i];
    int ownLabel = 0;
    for (int j = 0; j < length; ++j)
    {
      if(i != j)
      {
        int3 neighbor = points[j];
        int neighborIdx = j;
        int neighborLabel = labels[j];

        //check for neighborhood
        int distance = abs(p.x - neighbor.x) + abs(p.y - neighbor.y) + abs(p.z - neighbor.z);

        if(distance == 1)
        {
          //neighbor found
          if (ownLabel == 0 && neighborLabel == 0)
          {
            //none has a label yet
            ownLabel = label++;
            neighborLabel = ownLabel;
            labels[i] = ownLabel;
            labels[neighborIdx] = ownLabel;
          } else if (ownLabel == 0 && neighborLabel != 0)
          {
            //neighbor has a label
            ownLabel = neighborLabel;
            labels[i] = ownLabel;
          } else if (ownLabel != 0 && neighborLabel == 0)
          {
            //voxel has label but neighbor doesn't
            labels[neighborIdx] = ownLabel;
          } else if (ownLabel != 0 && neighborLabel != 0 && ownLabel != neighborLabel)
          {
            //both have a label
            int lowerLabel = ownLabel < neighborLabel ? ownLabel : neighborLabel;
            int upperLabel = ownLabel > neighborLabel ? ownLabel : neighborLabel;
            while (equalLabels.find(lowerLabel) != equalLabels.end())
            {
              lowerLabel = equalLabels.find(lowerLabel)->second;
            }
            equalLabels[lowerLabel] = upperLabel;
          }
        }
      }

    }
  }
  for (int k = 0; k < length; ++k)
  {
    //fix different labels
    int tmpLabel = labels[k];
    while(equalLabels.find(tmpLabel) != equalLabels.end())
    {
      tmpLabel = equalLabels.find(tmpLabel)->second;
    }
    labels[k] = tmpLabel;
  }

  return labels;
}


/**
 * Takes the int3 representation of the surface (approximation) and removes "small" regions
 * @param points coordinate represntation of isosurface voxel
 * @param dimensions isosurface dimensions
 * @param length number of voxel
 * @param newPts OUTPUT new set NOT containing small structures
 * @return number of points in the resulting set
 */
int Isosurface::removeSmallRegions(int3 *points, const int3 dimensions, const long length, const int threshold,
                                   int3 **newPts)
{
  //label regions
  int* labels = labelRegions(points, dimensions, length);

  //count points per region
  std::map<int,int> labelCounts;
  for (int i = 0; i < length; ++i)
  {
    labelCounts[labels[i]]++;
  }

  //remove small regions (threshold)
  int sum = 0;
  std::vector<int3> tmpPts = vector<int3>();
  for (int j = 0; j < length; ++j)
  {
    if(labelCounts[labels[j]] >= threshold)
    {
      sum += labelCounts[labels[j]];
      tmpPts.push_back(points[j]);
    }
  }

  //cleanup
  delete[] labels;
  int3* data = tmpPts.data();
  *newPts = (int3*) malloc(sum * sizeof(*newPts));
  memcpy(*newPts, data, sum * sizeof(*newPts));

  return sum;
}




