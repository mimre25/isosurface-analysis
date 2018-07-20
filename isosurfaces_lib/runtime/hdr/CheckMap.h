//
// Created by mimre on 20.06.18.
//

#ifndef ISOSURFACES_CHECKMAP_H
#define ISOSURFACES_CHECKMAP_H


#include "../../../../../../../../x86_64_linux/g/gcc/6.2.0/include/c++/6.2.0/string"

class CheckMap
{
  std::string fileName;
public:
  CheckMap(const std::string &fileName);

  int run();
};


#endif //ISOSURFACES_CHECKMAP_H
