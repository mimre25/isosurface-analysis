//
// Created by mimre on 1/18/17.
//

#ifndef ISOSURFACES_JSONHANDLER_H
#define ISOSURFACES_JSONHANDLER_H


#include "JSONObject.h"
#include "DAO/hdr/RunInformation.h"

namespace json
{
  class JSONHandler
  {
  public:
    template <class T> static void saveJSON(T object, std::string filename);
    template <class T> static T loadJSON(std::string filename);
  };
}

#endif //ISOSURFACES_JSONHANDLER_H
