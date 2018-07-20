//
// Created by mimre on 6/15/18.
//

#include <DAO/hdr/RunInformation.h>
#include <tuple>
#include <set>
#include "utils/hdr/RuntimeInfoUtils.h"



/**
 * creates a set with all entries in the hostInfo to have a quick lookup for already existing ones
 * @param hostInfo
 * @return
 */
std::set<entry> RuntimeInfoUtils::createHostSet(RunInformation& hostInfo)
{
  printf("Creating Host Set\n");
  std::set<entry> hostSet;
  printf("Generating %lu entries\n", hostInfo.similarityMaps.size());
  for (auto m : hostInfo.similarityMaps)
  {
    hostSet.insert(entry {m.var1, m.var2, m.t1, m.t2});
  }
  printf("Done Creating Host Set\n");
  return hostSet;
}


bool operator<(const entry& lhs, const entry& rhs)
{
// assumes there is a bool operator< for T
return std::tie(lhs.v1, lhs.v2, lhs.t1, lhs.t2) < std::tie(rhs.v1, rhs.v2, rhs.t1, rhs.t2);
}