//
// Created by mimre on 1/18/17.
//

#ifndef ISOSURFACES_REPRESENTATIVESINFO_H
#define ISOSURFACES_REPRESENTATIVESINFO_H


#include <string>
#include "json/hdr/JSONObject.h"

class RepresentativesInfo : json::JSONObject
{
public:

  RepresentativesInfo(int valueId, float isovalue, int repId, float importance, const std::string &filename, const int mapId);

  int valueId;
  float isovalue;
  int repId;
  RepresentativesInfo();

  float importance;
  std::string filename;
  int mapId;
};


#endif //ISOSURFACES_REPRESENTATIVESINFO_H
