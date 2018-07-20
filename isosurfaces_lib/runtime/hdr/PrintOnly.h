//
// Created by mimre on 1/20/17.
//

#ifndef ISOSURFACES_PRINTONLY_H
#define ISOSURFACES_PRINTONLY_H


#include "DAO/hdr/RunInformation.h"

class PrintOnly
{
public:
  void run();

  RunInformation runInformation;

  PrintOnly(const RunInformation &runInformation);

};


#endif //ISOSURFACES_PRINTONLY_H
