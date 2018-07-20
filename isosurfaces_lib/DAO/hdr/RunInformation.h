//
// Created by mimre on 1/18/17.
//

#ifndef ISOSURFACES_RUNINFORMATION_H
#define ISOSURFACES_RUNINFORMATION_H


#include <string>
#include <vector>

#include "VolumeInformation.h"
#include "SimilarityMapInformation.h"
#include "json/hdr/JSONObject.h"

class RunInformation : public json::JSONObject
{
public:


  std::vector<int> dimensions;
  std::vector<std::string> variables;
  int dfDownscale;
  std::vector<VolumeInformation> volumes;
  std::vector<SimilarityMapInformation> similarityMaps;

  RunInformation(const std::vector<int> &dimensions, const std::vector<std::string> &variables,
                 const std::vector<VolumeInformation> &volumens,
                 const std::vector<SimilarityMapInformation> &similarityMaps);

  RunInformation(const std::vector<int> &dimensions);

  RunInformation();

  void addVolumeInformation(std::string variable, int timestep, int numIsoValues,
                            const std::vector<RepresentativesInfo> &representatives, double runtime);

  void addSimilarityMap(std::string var1, std::string var2, int t1, int t2, std::string fileName);


  void addSimilarityMap(SimilarityMapInformation similarityMapInformation);
};



#endif //ISOSURFACES_RUNINFORMATION_H
