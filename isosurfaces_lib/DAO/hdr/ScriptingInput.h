//
// Created by mimre on 1/23/17.
//

#ifndef ISOSURFACES_SCRIPTINGINPUT_H
#define ISOSURFACES_SCRIPTINGINPUT_H


#include "DataInfo.h"

class ScriptingInput : public json::JSONObject
{

public:
  std::vector<DataInfo> entries;


  ScriptingInput(const std::vector<DataInfo> &entries);

  ScriptingInput();


};


#endif //ISOSURFACES_SCRIPTINGINPUT_H
