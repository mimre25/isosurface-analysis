//
// Created by mimre on 8/15/16.
//

#ifndef ISOSURFACES_SETTINGS_H
#define ISOSURFACES_SETTINGS_H

#include <string>
using namespace std;
class Settings
{
public:
  const int histogramSize;
  const int similarityMapSize;
  const int numberOfDistanceFields;
  const string outputFolder;
  const string inputFolder;
  const string fileName;

  Settings();

  const int dimX;
  const int dimY;
  const int dimZ;
  const float minValue;
  const float maxValue;
  const int scale;
  const bool uniform;
  const bool jhPar;
  const int numberOfSamples;
  const bool approximation;
  const int dfDownscale;



  Settings(const int histogramSize, const int similarityMapSize, const int numberOfDistanceFields,
           const string &outputFolder, const string &inputFolder, const string &fileName, const int dimX,
           const int dimY, const int dimZ, const float minValue, const float maxValue, const int scale,
           const bool uniform, const bool jhPar, const int numberOfSamples, const bool approximation, const int dfDownscale);
};


#endif //ISOSURFACES_SETTINGS_H
