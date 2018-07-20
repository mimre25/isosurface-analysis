//
// Created by mimre on 1/23/17.
//

#ifndef ISOSURFACES_SCRIPTING_H
#define ISOSURFACES_SCRIPTING_H

#define CRC false

#include "DAO/hdr/DataInfo.h"
#include "DAO/hdr/ScriptingInput.h"

class Scripting
{

  ScriptingInput input;

public:
  Scripting(const ScriptingInput &input);

  void run();

};


#endif //ISOSURFACES_SCRIPTING_H
