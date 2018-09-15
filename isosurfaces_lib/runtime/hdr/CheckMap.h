//
// Created by mimre on 20.06.18.
//

#ifndef ISOSURFACES_CHECKMAP_H
#define ISOSURFACES_CHECKMAP_H

#include <string>


class CheckMap
{
  std::string fileName;
public:
  CheckMap(const std::string &fileName);

  int run();
};


#endif //ISOSURFACES_CHECKMAP_H
