/*
 * marching_cubes.cpp
 *  MC Algorithm implemented by Paul Bourke wiki contribution
 *  Qt-Adaption Created on: 15.07.2009
 *      Author: manitoo
 */
/*
 * Edited by mimre in 2016
 *
 */


#include "types/hdr/MarchingCubes.h"
#include "filehandler/hdr/FileWriter.h"
#include "utils/hdr/StringUtils.h"
#include "types/hdr/DistanceField.h"
#include "types/hdr/JointHistogram.h"
#include "visualization/hdr/HistogramVisualizer.h"
#include "types/hdr/SimilarityMap.h"
#include "visualization/hdr/SimilarityMapVisualizer.h"
#include "utils/hdr/Report.h"
#include "types/hdr/Settings.h"
#include <boost/filesystem.hpp>
#include "isosurfaces_cuda/funcs.h"
#include "runtime/hdr/globals.h"


using namespace filehandler;
#include "sys/types.h"
#include "sys/sysinfo.h"

MarchingCubes::~MarchingCubes()
{}



/****************** HELPER FUNCTION FOR MARCHING CUBES ******************/
/*
fGetOffset finds the approximate point of intersection of the surface
between two points with the values fValue1 and fValue2
*/
float MarchingCubes::fGetOffset(const float &fValue1, const float &fValue2, const float &fValueDesired)
{
  float fDelta = fValue2 - fValue1;
  if (fDelta == 0.0)
  { return 0.5; }
  return (fValueDesired - fValue1) / fDelta;
}

void MarchingCubes::vNormalizeVector(GLvector &rfVectorResult, GLvector &rfVectorSource)
{
  float fOldLength;
  float fScale;

  fOldLength = sqrt((rfVectorSource.fX * rfVectorSource.fX) +
                    (rfVectorSource.fY * rfVectorSource.fY) +
                    (rfVectorSource.fZ * rfVectorSource.fZ));

  if (fOldLength == 0.0)
  {
    rfVectorResult.fX = rfVectorSource.fX;
    rfVectorResult.fY = rfVectorSource.fY;
    rfVectorResult.fZ = rfVectorSource.fZ;
  } else
  {
    fScale = 1.0 / fOldLength;
    rfVectorResult.fX = rfVectorSource.fX * fScale;
    rfVectorResult.fY = rfVectorSource.fY * fScale;
    rfVectorResult.fZ = rfVectorSource.fZ * fScale;
  }
}


//vGetNormal() finds the gradient of the scalar field at a point
//This gradient can be used as a very accurate vertx normal for lighting calculations
void MarchingCubes::vGetNormal(GLvector &rfNormal, const float &fX, const float &fY, const float &fZ)
{
  rfNormal.fX = datas(fX - 1.0, fY, fZ) - datas(fX + 1.0, fY, fZ);
  rfNormal.fY = datas(fX, fY - 1.0, fZ) - datas(fX, fY + 1.0, fZ);
  rfNormal.fZ = datas(fX, fY, fZ - 1.0) - datas(fX, fY, fZ + 1.0);
  vNormalizeVector(rfNormal, rfNormal);
}

//marchingCube performs the Marching Cubes algorithm on a single cube
void MarchingCubes::marchingCube(const float &fX, const float &fY, const float &fZ, const float &fTv,
                                 vector<float3> *currentPoints)
{
  float fScale = 1.0;
  int iCorner, iVertex, iVertexTest, iEdge, iTriangle, iFlagIndex, iEdgeFlags;
  float fOffset;
  float afCubeValue[8];
  GLvector asEdgeVertex[12];
  GLvector asEdgeNorm[12];

  //Make a local copy of the values at the cube's corners
  for (iVertex = 0;
       iVertex < 8;
       iVertex++)
  {
    afCubeValue[iVertex] = datas(fX + a2fVertexOffset[iVertex][0] * fScale,
                                            fY + a2fVertexOffset[iVertex][1] * fScale,
                                            fZ + a2fVertexOffset[iVertex][2] * fScale);
  }

  //Find which vertices are inside of the surface and which are outside
  iFlagIndex = 0;
  for (iVertexTest = 0; iVertexTest < 8; iVertexTest++)
  {
    if (afCubeValue[iVertexTest] <= fTv)
    { iFlagIndex |= 1 << iVertexTest; }
  }

  //Find which edges are intersected by the surface
  iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

  //If the cube is entirely inside or outside of the surface, then there will be no intersections
  if (iEdgeFlags == 0)
  {
    return;
  }

  //Find the point of intersection of the surface with each edge
  //Then find the normal to the surface at those points
  for (iEdge = 0; iEdge < 12; iEdge++)
  {
    //if there is an intersection on this edge
    if (iEdgeFlags & (1 << iEdge))
    {
      fOffset = fGetOffset(afCubeValue[a2iEdgeConnection[iEdge][0]], afCubeValue[a2iEdgeConnection[iEdge][1]],
                           fTv);

      asEdgeVertex[iEdge].fX = fX + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][0] +
                                     fOffset * a2fEdgeDirection[iEdge][0]) * fScale;
      asEdgeVertex[iEdge].fY = fY + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][1] +
                                     fOffset * a2fEdgeDirection[iEdge][1]) * fScale;
      asEdgeVertex[iEdge].fZ = fZ + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][2] +
                                     fOffset * a2fEdgeDirection[iEdge][2]) * fScale;

      vGetNormal(asEdgeNorm[iEdge], asEdgeVertex[iEdge].fX, asEdgeVertex[iEdge].fY, asEdgeVertex[iEdge].fZ);
    }
  }


  //Draw the triangles that were found.  There can be up to five per cube

  for (iTriangle = 0; iTriangle < 5; iTriangle++)
  {
    if (a2iTriangleConnectionTable[iFlagIndex][3 * iTriangle] < 0)
    { break; }

    for (iCorner = 0; iCorner < 3; iCorner++)
    {
      iVertex = a2iTriangleConnectionTable[iFlagIndex][3 * iTriangle + iCorner];

      float3 point = make_float3(asEdgeVertex[iVertex].fX, asEdgeVertex[iVertex].fY, asEdgeVertex[iVertex].fZ);
      float3 normal = make_float3(asEdgeNorm[iVertex].fX, asEdgeNorm[iVertex].fY, asEdgeNorm[iVertex].fZ);
      currentPoints->push_back(point);
      isosurface->ptsAndNormals.push_back(pair<float3, float3>(point, normal));
      isosurface->points.push_back(point);
    }
  }
}


float MarchingCubes::datas(const float &fX, const float &fY, const float &fZ)
{
  float result;
  int ix = (int) round(fX);
  int iy = (int) round(fY);
  int iz = (int) round(fZ);


  int dimX = isosurface->dimX;
  int dimY = isosurface->dimY;
  int dimZ = isosurface->dimZ;

  if (ix >= dimX)
  {
    ix = dimX - 1;
  } else if (ix < 0)
  {
    ix = 0;
  }
  if (iy >= dimY)
  {
    iy = dimY - 1;
  } else if (iy < 0)
  {
    iy = 0;
  }
  if (iz >= dimZ)
  {
    iz = dimZ - 1;
  } else if (iz < 0)
  {
    iz = 0;
  }
  result = isosurface->image[ix][iy][iz];
  return result;
}

/****************** END HELPER FUNCTION FOR MARCHING CUBES ******************/


MarchingCubes::MarchingCubes(Isosurface *isosurface) : isosurface(isosurface)
{}

