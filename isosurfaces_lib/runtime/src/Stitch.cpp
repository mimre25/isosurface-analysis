//
// Created by mimre on 6/15/18.
//

#include <DAO/hdr/RunInformation.h>
#include <json/hdr/JSONHandler.h>
#include <runtime/hdr/globals.h>
#include <set>
#include <tuple>
#include <utils/hdr/RuntimeInfoUtils.h>
#include "runtime/hdr/Stitch.h"

Stitch::Stitch(const std::vector<std::string> &files) : files(files)
{}



void stitch2(RunInformation& host, RunInformation& partial)
{
  for (auto& v: partial.volumes)
  {
    host.addVolumeInformation(partial.variables[v.variable], v.timestep, v.numIsovalues, v.representatives, v.runtime);
  }
  for (auto &m: partial.similarityMaps)
  {
    host.addSimilarityMap(partial.variables[m.var1], partial.variables[m.var2], m.t1, m.t2, m.filename);
  }
}

/**
 * stitches partial to host file, using a set to quickly look up if an entry exists
 * @param host
 * @param partial
 * @param hostSet
 */
void stitch(RunInformation& host, RunInformation& partial, std::set<entry>& hostSet)
{
  printf("Start stitching\n");
  printf("host set size:: %lu\n", hostSet.size());
  for(auto m : partial.similarityMaps)
  {
    entry e = entry {m.var1, m.var2, m.t1, m.t2};
    if(hostSet.find(e) == hostSet.end())
    {
      host.addSimilarityMap(m);
      hostSet.insert(e);
    }
  }
  printf("Stitching done\n");
}


/**
 * Stitches similarity maps in the jsonfile together
 * @return
 */
void Stitch::run()
{

  printf("Loading files\n");
  printf("%s\n", files[0].c_str());
  RunInformation hostFile = json::JSONHandler::loadJSON<RunInformation>(files[0]);

  std::vector<RunInformation> partials;
  for (int i = 1; i < files.size(); ++i)
  {
    printf("%s\n", files[i].c_str());
    partials.push_back(json::JSONHandler::loadJSON<RunInformation>(files[i]));
  }
  printf("File loading done\n");

  RuntimeInfoUtils riu;
  auto hostSet = riu.createHostSet(hostFile);

  printf("Hostset size: %lu\n", hostSet.size());

  for (int i = 0; i < partials.size(); ++i)
  {
    stitch2(hostFile, partials[i]);
  }
  printf("Hostset size after stitching %lu\n", hostSet.size());

  printf("Saving file\n");
  json::JSONHandler::saveJSON(hostFile, "stitched.json");
  printf("Done saving file\nEnjoy your result!\n");




}