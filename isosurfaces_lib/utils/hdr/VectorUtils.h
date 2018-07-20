//
// Created by mimre on 1/19/17.
//

#ifndef ISOSURFACES_VECTORUTILS_H
#define ISOSURFACES_VECTORUTILS_H

#include <glob.h>
#include <vector>
#include <algorithm>

class VectorUtils
{
public:
  template <class type> static size_t getElementIndex(const std::vector<type>& vec, type elem)
  {
    auto it = std::find(vec.begin(), vec.end(), elem);
    size_t index;
    if (it == vec.end())
    {
      index = (size_t) -1;
    } else
    {
      index = (size_t) std::distance(vec.begin(), it);
    }
    return index;
  }
};


#endif //ISOSURFACES_VECTORUTILS_H
