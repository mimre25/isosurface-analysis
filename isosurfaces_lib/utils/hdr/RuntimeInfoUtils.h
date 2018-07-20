//
// Created by mimre on 6/15/18.
//

#ifndef ISOSURFACES_RUNTIMEINFOUTILS_H
#define ISOSURFACES_RUNTIMEINFOUTILS_H

#include <set>

struct entry
{
  int v1;
  int v2;
  int t1;
  int t2;

};
bool operator<(const entry& lhs, const entry& rhs);


class RuntimeInfoUtils
{

public:
  std::set<entry> createHostSet(RunInformation &hostInfo);
};


#endif //ISOSURFACES_RUNTIMEINFOUTILS_H
