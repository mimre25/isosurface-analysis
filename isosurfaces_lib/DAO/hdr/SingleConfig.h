//
// Created by mimre on 2/3/17.
//

#include <string>
#include <vector>

#ifndef ISOSURFACES_SINGLECONFIG_H
#define ISOSURFACES_SINGLECONFIG_H


class SingleConfig
{
public:
  std::vector<int> dimensions;
  int timestep;
  std::string fileName;
  float minValue;
  float maxValue;
  std::string outputFolder;
  std::string variableName;
  int dfDownscale;
  std::string jsonFile;

  SingleConfig();


  SingleConfig(const std::string &fileName, const std::vector<int> &dimensions, float minValue, float maxValue);

};


#endif //ISOSURFACES_SINGLECONFIG_H

