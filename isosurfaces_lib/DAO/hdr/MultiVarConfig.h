//
// Created by mimre on 2/8/17.
//

#ifndef ISOSURFACES_MULTIVARCONFIG_H
#define ISOSURFACES_MULTIVARCONFIG_H


#include <string>
#include <vector>

class MultiVarConfig
{
public:
  std::string jsonFile;
  std::vector<int> timeSteps;
  std::string dataRoot;
  int dfDownscale;


  MultiVarConfig();
};


#endif //ISOSURFACES_MULTIVARCONFIG_H
