//
// Created by mimre on 3/31/17.
//

#ifndef ISOSURFACES_SURFACETOSURFACE_H
#define ISOSURFACES_SURFACETOSURFACE_H


#include "types/hdr/MarchingCubes.h"
#include "DAO/hdr/SingleConfig.h"
#include "types/hdr/DistanceField.h"

class SurfaceToSurface
{

  float* compute_isovalues(Settings *settings, float *data, int dataSize);

  Isosurface* isosurface;
  Isosurface* isosurface2;
  Settings*  settings;
  SingleConfig* config;
  MarchingCubes* marchingCubes;

public:
  static void calculateDistanceField(Settings *settings, SingleConfig *config, long length, int3 *points,
                                       long length2, int3 *points2, vector<DistanceField> *fields, int id= -1, int id2 = -1);
  static void calculateDistanceField(Settings *settings, SingleConfig *config, long length, int3 *points,
                                     long length2, int3 *points2, vector<vector<DistanceField>> *fields, int id1, int id2);
  SurfaceToSurface(Isosurface *isosurface, Isosurface *isosurface2, Settings *settings, SingleConfig *config);

  void run();
};


#endif //ISOSURFACES_SURFACETOSURFACE_H
