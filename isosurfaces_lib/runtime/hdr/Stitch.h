//
// Created by mimre on 6/15/18.
//

#ifndef ISOSURFACES_STITCH_H
#define ISOSURFACES_STITCH_H


#include <string>
#include <vector>

class Stitch
{
  std::vector<std::string> files;

public:
  Stitch(const std::vector<std::string> &files);

  void run();

};


#endif //ISOSURFACES_STITCH_H
