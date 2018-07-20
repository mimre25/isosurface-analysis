//
// Created by mimre on 1/18/17.
//

#ifndef ISOSURFACES_VOLUMEINFORMATION_H
#define ISOSURFACES_VOLUMEINFORMATION_H


#include <vector>
#include "json/hdr/JSONObject.h"
#include "RepresentativesInfo.h"

class VolumeInformation : public json::JSONObject
{
public:

  int variable;
  int timestep;

  VolumeInformation();

  VolumeInformation(int variable, int timestep, int numIsovalues, double runtime,
                    const std::vector<RepresentativesInfo> &representatives);

  int numIsovalues;
  double runtime;
  std::vector<RepresentativesInfo> representatives;

};


#endif //ISOSURFACES_VOLUMEINFORMATION_H
