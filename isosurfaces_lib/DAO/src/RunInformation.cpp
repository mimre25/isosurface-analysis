//
// Created by mimre on 1/18/17.
//

#include "DAO/hdr/RunInformation.h"
#include "utils/hdr/VectorUtils.h"

RunInformation::RunInformation(const std::vector<int> &dimensions, const std::vector<std::string> &variables,
                               const std::vector<VolumeInformation> &volumens,
                               const std::vector<SimilarityMapInformation> &similarityMaps) : dimensions(dimensions),
                                                                                              variables(variables),
                                                                                              volumes(volumens),
                                                                                              similarityMaps(
                                                                                                  similarityMaps)
{}

RunInformation::RunInformation(const std::vector<int> &dimensions) : dimensions(dimensions)
{}

void RunInformation::addVolumeInformation(std::string variable, int timestep, int numIsoValues,
                                          const std::vector<RepresentativesInfo> &representatives, double runtime)
{
  int var = (int) VectorUtils::getElementIndex(variables, variable);
  if (var == -1)
  {
    variables.push_back(variable);
    var = (int) VectorUtils::getElementIndex(variables, variable);
  }
  volumes.push_back(VolumeInformation(var, timestep, numIsoValues, runtime, representatives));
}

void RunInformation::addSimilarityMap(std::string var1, std::string var2, int t1, int t2, std::string fileName)
{
  int v1 = (int) VectorUtils::getElementIndex(variables, var1);
  if (v1 == -1)
  {
    variables.push_back(var1);
    v1 = (int) VectorUtils::getElementIndex(variables, var1);
  }
  int v2 = (int) VectorUtils::getElementIndex(variables, var2);
  if (v2 == -1)
  {
    variables.push_back(var2);
    v2 = (int) VectorUtils::getElementIndex(variables, var2);
  }

  similarityMaps.push_back(SimilarityMapInformation(v1,t1,v2,t2,fileName));
}

RunInformation::RunInformation()
{}

void RunInformation::addSimilarityMap(SimilarityMapInformation similarityMapInformation)
{
  similarityMaps.push_back(similarityMapInformation);
}


