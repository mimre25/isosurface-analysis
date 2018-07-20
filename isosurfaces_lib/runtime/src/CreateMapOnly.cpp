//
// Created by mimre on 6/13/17.
//

#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <types/hdr/Isosurface.h>
#include <runtime/hdr/globals.h>
#include <types/hdr/DistanceField.h>
#include <types/hdr/SimilarityMap.h>
#include <isosurfaces_cuda/funcs.h>
#include <runtime/hdr/DistanceEqualization.h>
#include <runtime/hdr/SurfaceToSurface.h>
#include "runtime/hdr/CreateMapOnly.h"

class FloatBinaryWriter;

using namespace std;

CreateMapOnly::CreateMapOnly(const RunInformation &runInformation) : runInformation(runInformation)
{}



void CreateMapOnly::run()
{
  string ds = "HURRICANE";
  string outputRoot = "~/tmp/DE/";
  string dataRoot = "/scratch/DATA-SCIVIS/DATA-SCALAR/Ionization-Vis-2006/";
  boost::filesystem::create_directories(outputRoot);
  int dimX = runInformation.dimensions[0];
  int dimY = runInformation.dimensions[1];
  int dimZ = runInformation.dimensions[2];
  int histogramSize = 128;


  SingleConfig config;
  config.dfDownscale = runInformation.dfDownscale;
  Settings settings(HIST_SIZE,
                    MAP_SIZE,
                    MAP_SIZE,
                    config.outputFolder,
                    DATAFOLDER,
                    config.fileName,
                    dimX,
                    dimY,
                    dimZ,
                    config.minValue,
                    config.maxValue,
                    SCALE,
                    true,
                    true,
                    NUM_SAMPLES,
                    true,
                    config.dfDownscale);



  int i = 0;
  for (VolumeInformation vI : runInformation.volumes)
  {

    ++i;
    if (i > 1)
    {
      continue;
    }
    Isosurface isosurface(dimX, dimY, dimZ, -1);
    string curVar = runInformation.variables[vI.variable];
    curVar = "GT";
    string filename = stringPrintf("%s/%s/%s%04d", dataRoot.c_str(), curVar.c_str(), curVar.c_str(),vI.timestep);
    cout << filename << endl;
    isosurface.loadFile(filename, false);

    float isovalues[vI.numIsovalues];
    cout << filename << endl;
    isosurface.loadFile(filename, false);//false means it's read as float
    string simMapFolder = runInformation.variables[vI.variable] +"-" + to_string(vI.timestep) + "/";

    if(SURFACE_TO_SURFACE_DISTANCE)
    {
      vector<int3*> pts;
      vector<long> lengths;
      for (RepresentativesInfo rI: vI.representatives)
      {
        pts.push_back(NULL);
        long length = isosurface.calculateSurfacePoints(true, rI.isovalue, &pts.back());
        lengths.push_back(length);
      }

      vector<vector<DistanceField>> fields;
      fields.resize((unsigned long) vI.numIsovalues);
      vector<vector<float>> distanceMatrix;
      distanceMatrix.resize((unsigned long) vI.numIsovalues);
      for (int k = 0; k < pts.size(); ++k)
      {
        fields[k].resize((unsigned long) vI.numIsovalues);
        distanceMatrix[k].resize((unsigned long) vI.numIsovalues);
        distanceMatrix[k][k] = 0;
      }
      for (int i = 0; i < pts.size(); ++i)
      {

        for (int j = i+1; j < pts.size(); ++j)
        {

          SurfaceToSurface::calculateDistanceField(&settings, &config, lengths[i], pts[i], lengths[j], pts[j], &fields, i, j);
          DistanceField field1 = fields[i][j];
          DistanceField field2 = fields[j][i];
          float acc = 0;
          for (int u = 0; u < field1.getNumberOfSamples(); ++u)
          {
            acc += field1.getDistancePtr()[u];
          }
          float m1 = acc/ field1.getNumberOfSamples();
          acc = 0;
          for (int u = 0; u < field2.getNumberOfSamples(); ++u)
          {
            acc += field2.getDistancePtr()[u];
          }
          float m2 = acc/ field2.getNumberOfSamples();
          distanceMatrix[i][j] = (m1+m2)/2;
          distanceMatrix[j][i] = (m1+m2)/2;
        }
      }
      SimilarityMap similarityMap(distanceMatrix, vI.numIsovalues);
      boost::filesystem::create_directories(outputRoot + simMapFolder);
      similarityMap.save(outputRoot + simMapFolder + "map.dat");
      cout << outputRoot + simMapFolder << endl;
    } else
    {
      vector<DistanceField> fields;

      int v = 0;
      for (RepresentativesInfo rI: vI.representatives)
      {
        float curIsovalue = rI.isovalue;
        curIsovalue = isovalues[v++];
        int curId = rI.valueId;
        int d = curId;
        isovalues[d] = curIsovalue;
        string calc = "calculating representative isosurface nr " + to_string(curId) + "/" +
                      to_string(vI.numIsovalues);
        cout << calc << endl;

        int3 *points;
        long length = isosurface.calculateSurfacePoints(true, curIsovalue, &points);


        //Distancefield compuation

        if (length > 0)
        {

          fields.push_back(
              DistanceField(points, runInformation.dfDownscale, dimX / runInformation.dfDownscale,
                            dimY / runInformation.dfDownscale, dimZ / runInformation.dfDownscale, dimX, dimY, dimZ,
                            length));


          fields.back().calculateDistanceField(false, true, 1500);
          fields.back().id = d;
        } else
        {
          cout << "\n" << d << ": empty\n";
        }


      }
      SimilarityMap similarityMap = calculate_histogram_W(fields, (dimX / runInformation.dfDownscale) *
                                                                  (dimY / runInformation.dfDownscale) *
                                                                  (dimZ / runInformation.dfDownscale), histogramSize,
                                                          fields.size(), false);
      similarityMap.setIsovalues(isovalues);

      boost::filesystem::create_directories(outputRoot + simMapFolder);
      similarityMap.save(outputRoot + simMapFolder + "map.dat");
      similarityMap.print();
      cout << outputRoot + simMapFolder << endl;

    }

    isosurface.clear();
    isosurface.freeImg();
  }
}
