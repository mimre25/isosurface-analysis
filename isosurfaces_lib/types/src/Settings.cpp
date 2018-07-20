//
// Created by mimre on 8/15/16.
//

#include "types/hdr/Settings.h"

Settings::Settings(const int histogramSize, const int similarityMapSize, const int numberOfDistanceFields,
                   const string &outputFolder, const string &inputFolder, const string &fileName, const int dimX,
                   const int dimY, const int dimZ, const float minValue, const float maxValue, const int scale,
                   const bool uniform, const bool jhPar, const int numberOfSamples, const bool approximation, const int dfDownscale)
    : histogramSize(histogramSize), similarityMapSize(similarityMapSize),
      numberOfDistanceFields(numberOfDistanceFields), outputFolder(outputFolder), inputFolder(inputFolder),
      fileName(fileName), dimX(dimX), dimY(dimY), dimZ(dimZ), minValue(minValue), maxValue(maxValue), scale(scale),
      uniform(uniform), jhPar(jhPar), numberOfSamples(numberOfSamples), approximation(approximation), dfDownscale(dfDownscale)
{}

