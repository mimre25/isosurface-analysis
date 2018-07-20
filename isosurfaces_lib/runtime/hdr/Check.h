//
// Created by mimre on 6/15/18.
//

#ifndef ISOSURFACES_CHECK_H
#define ISOSURFACES_CHECK_H


#include <string>

class Check
{
  std::string file;
  int ts1;
  int ts2;

public:
  Check(const std::string &file, int ts1, int ts2);

  int run();
};


#endif //ISOSURFACES_CHECK_H
