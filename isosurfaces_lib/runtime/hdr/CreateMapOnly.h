//
// Created by mimre on 6/13/17.
//

#ifndef ISOSURFACES_CREATEMAPONLY_H
#define ISOSURFACES_CREATEMAPONLY_H


#include <DAO/hdr/RunInformation.h>

class CreateMapOnly
{

  RunInformation runInformation;

public:
  CreateMapOnly(const RunInformation &runInformation);
  void run();
};


#endif //ISOSURFACES_CREATEMAPONLY_H
