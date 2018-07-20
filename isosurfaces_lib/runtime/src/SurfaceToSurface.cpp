//
// Created by mimre on 3/31/17.
//

#include <runtime/hdr/DistanceEqualization.h>
#include "runtime/hdr/SurfaceToSurface.h"
#include "utils/hdr/MemUtils.h"
#include "utils/hdr/StringUtils.h"
#include "types/hdr/DistanceField.h"
#include "utils/hdr/Report.h"



void SurfaceToSurface::run()
{
  MemUtils::checkmem("beginning");
  float curIsovalue;    //Targetvalue
  float curIsovalue2;    //Targetvalue
  int numDistanceFields = settings->numberOfDistanceFields;
  string fileName = settings->fileName;



  int dimX = isosurface->dimX;
  int dimY = isosurface->dimY;
  int dimZ = isosurface->dimZ;
  //get min max and interval values

  vector<string> fileparts = utils::StringUtils::split(fileName, '/');
  string fname = fileparts.back();
  vector<DistanceField> fields;
  string total = "total";
  string isosurfacesTotal = "IsoSurfaces & Distancefields Total";



  //find isovalues
  Report::begin(total);
  float* isovalues = compute_isovalues(settings, isosurface->imageAsFloatPtr(), dimX * dimY * dimZ);
  float* isovalues2 = compute_isovalues(settings, isosurface2->imageAsFloatPtr(), dimX * dimY * dimZ);

  Report::begin(isosurfacesTotal);

  //Start Computation
  bool approx = true;



  string comp = "computing isosurface (marching || approx)";

  int3 *points = NULL;
  int3 *points2 = NULL;
  MemUtils::checkmem("algo start");
  vector<int> possibleSurfaces;

  for (int d = 0; d < numDistanceFields-1; ++d)
  {
    MemUtils::checkmem("loop start");

    long length = 0;
    long length2 = 0;

    curIsovalue = isovalues[d];
    curIsovalue2 = isovalues2[d+1];


    Report::begin(comp);
    cout << " nr " << to_string(d) << "/" << to_string(numDistanceFields-1) << endl;
    MemUtils::checkmem("begin approx");
    length = isosurface->calculateSurfacePoints(approx, curIsovalue, &points);
    MemUtils::checkmem("end approx");
    Report::end(comp);
    length2 = isosurface2->calculateSurfacePoints(approx, curIsovalue2, &points2);


    //Distancefield compuation

    MemUtils::checkmem("begin df");
    if (length > 0 && length2 > 0)
    {

      calculateDistanceField(settings, config, length, points, length2, points2, &fields);

    } else
    {
      cout << "\n" << d << ": empty\n";
    }
    MemUtils::checkmem("end df");
    isosurface->clear();
    isosurface2->clear();
    MemUtils::checkmem("loop end");

    if(points != NULL)
    {
      free(points);
    }
    if(points2 != NULL)
    {
      free(points2);
    }
  }
  MemUtils::checkmem("algo end");

  isosurface->freeImg();
  isosurface2->freeImg();

}


float* SurfaceToSurface::compute_isovalues(Settings *settings, float *data, int dataSize)
{
  cout<< "Sorting values" << endl;
  int histSize = settings->numberOfDistanceFields;

  float* isovalues = (float *) malloc(histSize * sizeof(float));

  float* vals = new float[dataSize];
  memcpy(vals, data, (size_t) (dataSize * sizeof(*data)));
  std::sort(&vals[0], &vals[dataSize-1]);
  int step = dataSize/histSize;
  for(int i = 0; i < histSize; ++i)
  {
    isovalues[i] = vals[step * i];
  }
  delete[] vals;

  return isovalues;
}

void SurfaceToSurface::calculateDistanceField(Settings *settings, SingleConfig *config, long length, int3 *points,
                                              long length2, int3 *points2, vector<vector<DistanceField>> *fields, int id, int id2)

{
  Report::begin(DistanceEqualization::currentStage + DistanceEqualization::DISTANCEFIELD);
  int dimX = settings->dimX;
  int dimY = settings->dimY;
  int dimZ = settings->dimZ;

  (*fields)[id][id2] = (
      DistanceField(points, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ,
                    length));
  (*fields)[id][id2].id = id;

  (*fields)[id][id2].calculateSurfaceToSurfaceDistanceField(settings->numberOfSamples, points2, length2);


  (*fields)[id2][id] =
      DistanceField(points2, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ,
                    length2);
  (*fields)[id2][id].id = id2;

  (*fields)[id2][id].calculateSurfaceToSurfaceDistanceField(settings->numberOfSamples, points, length);

  Report::end(DistanceEqualization::currentStage + DistanceEqualization::DISTANCEFIELD);
}


void SurfaceToSurface::calculateDistanceField(Settings *settings, SingleConfig *config, long length, int3 *points,
                                              long length2, int3 *points2, vector<DistanceField> *fields, int id, int id2)
{
  Report::begin(DistanceEqualization::currentStage + DistanceEqualization::DISTANCEFIELD);
  int dimX = settings->dimX;
  int dimY = settings->dimY;
  int dimZ = settings->dimZ;

  fields->push_back(
      DistanceField(points, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ,
                    length));
  fields->back().id = id;

  fields->back().calculateSurfaceToSurfaceDistanceField(settings->numberOfSamples, points2, length2);


  fields->push_back(
      DistanceField(points2, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ,
                    length2));
  fields->back().id = id2;

  fields->back().calculateSurfaceToSurfaceDistanceField(settings->numberOfSamples, points, length);

  Report::end(DistanceEqualization::currentStage + DistanceEqualization::DISTANCEFIELD);
}

SurfaceToSurface::SurfaceToSurface(Isosurface *isosurface, Isosurface *isosurface2, Settings *settings,
                                   SingleConfig *config)
{

  this->settings = settings;
  this->config = config;
  this->isosurface = isosurface;
  this->isosurface2 = isosurface2;
  marchingCubes = new MarchingCubes(isosurface);
}
