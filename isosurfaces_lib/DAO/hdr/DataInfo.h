//
// Created by mimre on 1/23/17.
//

#ifndef ISOSURFACES_DATAINFO_H
#define ISOSURFACES_DATAINFO_H


#include <vector>
#include <string>
#include "json/hdr/JSONObject.h"
#include "VariableInfo.h"

class DataInfo : public json::JSONObject
{
public:
  std::vector<int> dimensions;
  std::vector<VariableInfo> variables;
  std::vector<int> timesteps;
  int dfDownscale;
  std::string fileFormatString;
  std::string inputFolder;
  std::string outputFolder;
  std::string jsonFile;

  DataInfo();

};





#endif //ISOSURFACES_DATAINFO_H
