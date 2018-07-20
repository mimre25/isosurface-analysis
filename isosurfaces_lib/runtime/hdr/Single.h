//
// Created by mimre on 1/13/17.
//

#ifndef ISOSURFACES_SINGLE_H
#define ISOSURFACES_SINGLE_H


#include "types/hdr/Isosurface.h"
#include "types/hdr/Settings.h"
#include "types/hdr/MarchingCubes.h"
#include "types/hdr/DistanceField.h"
#include "types/hdr/SimilarityMap.h"
#include "DAO/hdr/RepresentativesInfo.h"
#include "DAO/hdr/SingleConfig.h"

struct PrintInformation
{
  int leadingZeros;
  string fileName;
};

class Single
{
public:
  void run();
  void clean();

private:
  Isosurface* isosurface;
  Settings*  settings;
  SingleConfig* config;
  MarchingCubes* marchingCubes;
  float* compute_isovalues(Settings *settings, float *data, int dataSize, bool uniform);
  void get_uniform_histogram(int* hist, int UNIFORM_HISTOGRAM_BIN_COUNT, float* data, float dataSize, Settings settings, int histSize);

  void init();
  void createFolder();
  void printIsosurface(string fileName);
  SimilarityMap calculateHistograms(vector<DistanceField> fields);
  void printSimilarityMap(const int mapSize, SimilarityMap* similarityMap, vector<int>& possibleValues);

  void printResults(unordered_map<int, SimilarityMap::RepInfo> *recommendedVals, float *isovalues, string fname,
                      vector<RepresentativesInfo> &repInfo, vector<DistanceField> &fields);
  void calculateDistanceField(Isosurface *isosurface, Settings *settings, long length, int3 *points,
                              vector<DistanceField> *fields);
  void OutputIsosurfaceAndDistancefield(string fname, int index, float curIsovalue, PrintInformation printInfo,
                                        DistanceField *field);
  void saveRunInfo(vector<RepresentativesInfo> repInfo, int mapSize, double runtime);
public:
  Single(Isosurface *isosurface, Settings *settings, SingleConfig* config);

  bool checkIfAlreadyProcessed();
  string histogramFolder;
  string isoSurfaceFolder;
  string outputFolder;
  string simMapFolder;
  string distanceFieldFolder;
  string logFolder;
  string logFile;
};


#endif //ISOSURFACES_SINGLE_H
