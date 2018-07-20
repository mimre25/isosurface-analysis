//
// Created by mimre on 2/22/17.
//

#ifndef ISOSURFACES_VARIABLEINFO_H
#define ISOSURFACES_VARIABLEINFO_H


#include <string>

class VariableInfo
{
public:
  std::string name;
  float minValue;
  float maxValue;
  std::string variableFolder;

  VariableInfo();
};


#endif //ISOSURFACES_VARIABLEINFO_H
