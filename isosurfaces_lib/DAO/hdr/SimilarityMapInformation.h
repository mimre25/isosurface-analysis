//
// Created by mimre on 1/18/17.
//

#ifndef ISOSURFACES_SIMILARITYMAPINFORMATION_H
#define ISOSURFACES_SIMILARITYMAPINFORMATION_H


#include <string>
#include "json/hdr/JSONObject.h"

class SimilarityMapInformation : public json::JSONObject
{
public:
  SimilarityMapInformation(int var1, int t1, int var2, int t2, const std::string &filename);

public:
  int var1;
  int t1;

  SimilarityMapInformation();

  int var2;
  int t2;
  std::string filename;


};


#endif //ISOSURFACES_SIMILARITYMAPINFORMATION_H
