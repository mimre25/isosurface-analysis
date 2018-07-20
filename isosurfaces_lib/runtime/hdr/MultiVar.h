//
// Created by mimre on 11/21/16.
//

#ifndef ISOSURFACES_MULTIVAR_H
#define ISOSURFACES_MULTIVAR_H


static const char *const DISTANCEFIELD_PREFIX = "distancefield-";

static const char *const DISTANCEFIELD_FOLDER = "/distancefields/";

#include "types/hdr/DistanceField.h"
#include "DAO/hdr/VolumeInformation.h"
#include "DAO/hdr/SimilarityMapInformation.h"
#include "DAO/hdr/RunInformation.h"
#include "DAO/hdr/MultiVarConfig.h"

class MultiVar
{
private:
  const string jsonFile;

  int dimX;
  int dimY;
  int dimZ;

public:
  MultiVar(const string &jsonFile);
  int parNum;
  string computeMultiMap(const string var1, const string var2, const int t1, const int t2, const MultiVarConfig &config,
                           const string prefix);

  void run();
};


#endif //ISOSURFACES_MULTIVAR_H
