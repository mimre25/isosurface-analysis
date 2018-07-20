//
// Created by mimre on 6/21/18.
//

#ifndef ISOSURFACES_HISTOGRAM_H
#define ISOSURFACES_HISTOGRAM_H


#include <string>
#include <vector>

class Histogram
{
  std::vector<std::string> files;
  int dimX, dimY, dimZ;
public:
  Histogram(const vector<string> &files, int dimX, int dimY, int dimZ);

  int run();
};


#endif //ISOSURFACES_HISTOGRAM_H
