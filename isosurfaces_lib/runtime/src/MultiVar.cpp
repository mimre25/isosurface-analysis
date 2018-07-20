//
// Created by mimre on 11/21/16.
//

#include <sstream>
#include <iostream>
#include <ThorSerialize/Traits.h>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <set>
#include "runtime/hdr/MultiVar.h"
#include "isosurfaces_cuda/funcs.h"
#include "visualization/hdr/SimilarityMapVisualizer.h"
#include "runtime/hdr/globals.h"
#include "DAO/hdr/RunInformation.h"
#include "types/hdr/VariableNameAnalyzer.h"
#include "json/hdr/JSONHandler.h"
#include "DAO/hdr/MultiVarConfig.h"
#include "runtime/hdr/DistanceEqualization.h"

struct stringEntry
{
  std::string v1;
  std::string v2;
  int t1;
  int t2;
};

bool checkIfAlreadyCompute(std::string v1, std::string v2, int t1, int t2, RunInformation runInfo, std::set<stringEntry>& hostSet)
{
  bool retVal = false;
//  auto maps = runInfo.similarityMaps;
//  for (int i = 0; !retVal && i < maps.size(); ++i)
//  {
//    auto m = maps[i];
//    if (m.var1 == v1 &&
//        m.var2 == v2 &&
//        m.t1 == t1 &&
//        m.t2 == t2)
//    {
//      retVal = true;
//    }
//  }
    retVal = hostSet.find(stringEntry { v1, v2, t1, t2}) != hostSet.end();

  return retVal;
}

bool operator<(const stringEntry& lhs, const stringEntry& rhs)
{
  // assumes there is a bool operator< for T
  return std::tie(lhs.v1, lhs.v2, lhs.t1, lhs.t2) < std::tie(rhs.v1, rhs.v2, rhs.t1, rhs.t2);
}



/**
 * creates a set with all entries in the hostInfo to have a quick lookup for already existing ones
 * @param hostInfo
 * @return
 */
std::set<stringEntry> createHostSetStr(RunInformation& hostInfo)
{
  printf("Creating Host Set\n");
  std::set<stringEntry> hostSet;
  printf("Generating %lu entries\n", hostInfo.similarityMaps.size());
  auto varNames = hostInfo.variables;
  for (auto m : hostInfo.similarityMaps)
  {
    hostSet.insert(stringEntry {varNames[m.var1], varNames[m.var2], m.t1, m.t2});
  }
  printf("Done Creating Host Set\n");
  return hostSet;
}

void MultiVar::run()
{
  cout << "reading json" << endl;
  MultiVarConfig config = json::JSONHandler::loadJSON<MultiVarConfig>(jsonFile);
  cout << "json read" << endl;
  string dataJson = config.dataRoot + "/" + config.jsonFile;

  cout << "datajson: " << dataJson << endl;
  RunInformation runInformation = json::JSONHandler::loadJSON<RunInformation>(dataJson);
  auto hostSet = createHostSetStr(runInformation);

  const int numVars = (const int) runInformation.variables.size();
  const int numTs = (const int) config.timeSteps.size();
  this->dimX = runInformation.dimensions[0];
  this->dimY = runInformation.dimensions[1];
  this->dimZ = runInformation.dimensions[2];
  parNum = config.timeSteps[0];
  RunInformation writeInfo;
  std::string outputFile;
  

  writeInfo = runInformation;
  outputFile = dataJson;

  //use this only if multiple processors run the same input file
//  outputFile = config.dataRoot + "/" + std::to_string(parNum) + ".json";
//  if ((access(outputFile.c_str(), F_OK) != -1))
//  {
//    writeInfo = json::JSONHandler::loadJSON<RunInformation>(outputFile);
//    auto writeSet = createHostSetStr(writeInfo);
//    hostSet.insert(writeSet.begin(), writeSet.end());
//  }
  string prefix = config.dataRoot + "-multi-tmp/";
  cout << "loop start" << endl;
  for (int i = 0; i < numVars; ++i)
  {
    string v1 = runInformation.variables[i];
    for (int j = i; j < numVars; ++j)
    {
      string v2 = runInformation.variables[j];
      for (int k = 0; k < numTs; ++k)
      {
        int t1 = config.timeSteps[k];
        string folder = prefix + v1 + "-" + to_string(t1);
        boost::filesystem::create_directories(folder);
        //Compute var x map (eg vort x chi with same timestep)
        if (i != j)
        {
          int t2 = t1;
          if(!checkIfAlreadyCompute(v1,v2,t1,t2, writeInfo, hostSet))
          {
            string mapFileName = computeMultiMap(v1, v2, t1, t2, config, folder);
            if (mapFileName != "failed")
            {
              writeInfo.addSimilarityMap(v1, v2, t1, t2, mapFileName);
              hostSet.insert(stringEntry{v1,v2,t1,t2});

            }
            cout << mapFileName << endl;
          }
        } else // different timestep, same variable (eg vort-1 x vort-2)
        {
          for (int l = k + 1; l < numTs; ++l)
          {
            int t2 = config.timeSteps[l];
            if (!checkIfAlreadyCompute(v1, v2, t1, t2, writeInfo, hostSet))
            {
              string mapFileName = computeMultiMap(v1, v2, t1, t2, config, folder);
              if (mapFileName != "failed")
              {
                writeInfo.addSimilarityMap(v1, v2, t1, t2, mapFileName);
                hostSet.insert(stringEntry{v1,v2,t1,t2});
              }
              cout << mapFileName << endl;
            }
          }
          json::JSONHandler::saveJSON(writeInfo, outputFile);

        }


      }
    }
  }
  json::JSONHandler::saveJSON(writeInfo, outputFile);


}


MultiVar::MultiVar(const string &jsonFile) : jsonFile(jsonFile)
{
}

string MultiVar::computeMultiMap(const string var1, const string var2, const int t1, const int t2,
                                 const MultiVarConfig &config,
                                 const string prefix)
{
  cout << "computation starts" << endl;
  clock_t start = clock();
  vector<DistanceField> fields1;
  vector<DistanceField> fields2;
  //load distancefields 1
  //load distancefields 2
  int size = (dimX / config.dfDownscale) * (dimY / config.dfDownscale) * (dimZ / config.dfDownscale);

  string prefix1 = config.dataRoot + "/" + var1 + "-" + to_string(t1) + DISTANCEFIELD_FOLDER;
  string prefix2 = config.dataRoot + "/" + var2 + "-" + to_string(t2) + DISTANCEFIELD_FOLDER;


  string filePrefix1 = prefix1 + DISTANCEFIELD_PREFIX;
  string filePrefix2 = prefix2 + DISTANCEFIELD_PREFIX;

  cout << prefix1 << endl << prefix2 << endl;

  if (!boost::filesystem::exists(prefix1)
      || !boost::filesystem::exists(prefix2))
  {
    return "failed";
  }


  for (int i = 0; i < MULTI_MAP_DF_NUMBER; ++i)
  {
    DistanceField df1 = DistanceField(config.dfDownscale, dimX, dimY, dimZ);
    DistanceField df2 = DistanceField(config.dfDownscale, dimX, dimY, dimZ);
    df1.loadFromFile(filePrefix1 + to_string(i) + DF_EXT, size);
    df2.loadFromFile(filePrefix2 + to_string(i) + DF_EXT, size);
    fields1.push_back(df1);
    fields2.push_back(df2);
  }
  fields1.insert(fields1.end(), fields2.begin(), fields2.end());


  SimilarityMap sm = calculate_histogram_W(fields1, size, HIST_SIZE, MULTI_MAP_DF_NUMBER, true);
  float isovs[MULTI_MAP_DF_NUMBER];
  vector<int> vals;
  for (int j = 0; j < MULTI_MAP_DF_NUMBER; ++j)
  {
    isovs[j] = j;
    vals.push_back(j);
  }
  sm.setIsovalues(isovs);
  stringstream stream, stream2;


  string outPutfile = stringPrintf("%s/TMP-multiMap-%d-%s-%d-%s-%d", prefix.c_str(), MULTI_MAP_DF_NUMBER, var1.c_str(), t1,
                                   var2.c_str(), t2);
  if (RESULTS_OUT)
  {
    sm.save(outPutfile + ".dat");
  }


  clock_t end = clock();
  double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
  std::ofstream outfile;

  outfile.open(config.dataRoot + "-multi-tmp/runtimes-tmp.csv", std::ios_base::app);
  outfile << outPutfile << "," << to_string(elapsed_secs) << endl;
  outfile.close();

  for (DistanceField &field : fields1)
  {
    if (field.isDistancePtrSet())
    {
      free(field.getDistancePtr());

    }
  }

  return outPutfile + ".dat";

}

