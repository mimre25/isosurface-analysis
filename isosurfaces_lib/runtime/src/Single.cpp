//
// Created by mimre on 1/13/17.
//

#include <boost/filesystem/operations.hpp>
#include <malloc.h>
#include <utils/hdr/FileUtils.h>
#include "runtime/hdr/Single.h"
#include "utils/hdr/StringUtils.h"
#include "utils/hdr/Report.h"
#include "isosurfaces_cuda/funcs.h"
#include "visualization/hdr/SimilarityMapVisualizer.h"
#include "DAO/hdr/RunInformation.h"
#include "json/hdr/JSONHandler.h"
#include "types/hdr/VariableNameAnalyzer.h"
#include "filehandler/hdr/FloatBinaryWriter.h"
#include "utils/hdr/MemUtils.h"

using namespace std;
using namespace filehandler;

#define NUM_DISTANCES 7


void reportArray2(ofstream &outfile, float *array, int arraySize)
{
  for (int i = 0; i < arraySize; ++i)
  {
    outfile << std::setprecision(32) << array[i] << ",";
  }
}


void reportDistances(float avgDistance, float maxError, float avgError,
                     float *distances, float stdDev, float normalizedMaxError,
                     float normalizedAvgError, std::string varFolder, std::string dataset)
{
  string fileName = varFolder + "/";

  fileName += "distances.csv";
  bool fileExists = FileUtils::fileExists(fileName);
  std::ofstream outfile;

  outfile.open(fileName, std::ios_base::app);

  if (!fileExists)
  {

    outfile << "dataset,";
    for (int i = 0; i < NUM_DISTANCES; ++i)
    {
      outfile << "d" << to_string(i) << ",";
    }


    outfile << "avgD" << "," << "maxError"
            << "," << "averageError" << "," << "stdDev" << ","
            << "normMaxE" << "," << "normAvgE" << endl;
  }

  outfile << dataset << ",";
  reportArray2(outfile, distances, NUM_DISTANCES);


  outfile << avgDistance << "," << maxError << "," << avgError << "," << stdDev << "," << normalizedMaxError << "," << normalizedAvgError << endl;
  outfile.close();
}





void evaluateDistances(float *distances, std::string varFolder, std::string dataset)
{

  float acc = 0;
  float maxD = -1e10;
  for (int j = 0; j < NUM_DISTANCES; ++j)
  {
    acc += distances[j];
    if (distances[j] > maxD)
    {
      maxD = distances[j];
    }
  }
  float avgDistance = acc/NUM_DISTANCES;


  float errors[NUM_DISTANCES];
  float accV = 0;
  acc = 0;
  float maxError = -1;

  for (int k = 0; k < NUM_DISTANCES; ++k)
  {
    errors[k] = abs(distances[k]-avgDistance);
    accV += errors[k] * errors[k];
    acc += errors[k];

    if (errors[k] > maxError)
    {
      maxError = errors[k];
    }
  }
  float avgError = acc/NUM_DISTANCES;
  float variance = accV/NUM_DISTANCES;
  float stdDev = sqrt(variance);

  float normalizedMaxError = maxError/avgDistance;
  float normalizedAvgError = avgError/avgDistance;
  reportDistances( avgDistance, maxError, avgError, distances, stdDev, normalizedMaxError, normalizedAvgError,varFolder, dataset);

}

bool Single::checkIfAlreadyProcessed()
{
  bool retVal = false;
  RunInformation runInformation;
  int curTS = config->timestep;
  auto curV = config->variableName;
  string jsonFile = outputFolder + config->jsonFile;
  if (( access( jsonFile.c_str(), F_OK ) != -1 ))
  {
    cout << jsonFile << endl;
    runInformation = json::JSONHandler::loadJSON<RunInformation>(jsonFile);

    for (int i = 0; i < runInformation.volumes.size(); ++i)
    {
      if(runInformation.variables[runInformation.volumes[i].variable] == curV && 
	runInformation.volumes[i].timestep == curTS)
      {
        retVal = true;
      }
    }

  }
  return retVal;
}




void Single::run()
{
  if(checkIfAlreadyProcessed())
  {
    printf("ts %d already processed... skipping", config->timestep);
    return;
  }
  MemUtils::checkmem("beginning");
  float curIsovalue;    //Targetvalue
  int mapSize = settings->similarityMapSize;
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
  float* isovalues = compute_isovalues(settings, isosurface->imageAsFloatPtr(), dimX * dimY * dimZ, settings->uniform);

  Report::begin(isosurfacesTotal);
  PrintInformation printInfo { numDistanceFields > 100 ? 3 : numDistanceFields > 10 ? 2 : 1, ""};
  //Start Computation
  bool approx = settings->approximation;

  for (int i = 0; i < mapSize; ++i)
  {
    printf("%f\n", isovalues[i]);
  }


  string comp = "computing isosurface (marching || approx)";

  int3 *points = NULL;
  MemUtils::checkmem("algo start");
  vector<int> possibleSurfaces;


  for (int d = 0; d < numDistanceFields; ++d)
  {
    MemUtils::checkmem("loop start");

    long length = 0;

    curIsovalue = isovalues[d];


    Report::begin(comp);
    cout << " nr " << to_string(d) << "/" << to_string(numDistanceFields-1) << endl;
    MemUtils::checkmem("begin approx");
    length = isosurface->calculateSurfacePoints(approx, curIsovalue, &points);
    MemUtils::checkmem("end approx");
    Report::end(comp);
    printf("Length: %lu\n",length);
    //Distancefield compuation

    MemUtils::checkmem("begin df");
    if (length > 0)
    {
      possibleSurfaces.push_back(d);
      calculateDistanceField(isosurface, settings, length, points, &fields);
      OutputIsosurfaceAndDistancefield(fname, d, curIsovalue, printInfo, &fields.back());
      fields.back().id = d;

    } else
    {
      cout << "\n" << d << ": empty\n";
    }
    MemUtils::checkmem("end df");
    isosurface->clear();
    MemUtils::checkmem("loop end");

    if(points != NULL)
    {
      free(points);
    }
  }
  MemUtils::checkmem("algo end");



  MemUtils::checkmem("map start");

  Report::end(isosurfacesTotal);

  mapSize = (int) fields.size();

  printf("starting simmap (jh) \n");
  SimilarityMap similarityMap = calculateHistograms(fields);
  similarityMap.setIsovalues(isovalues);

  printSimilarityMap(mapSize, &similarityMap, possibleSurfaces);

  unordered_map<int, SimilarityMap::RepInfo> *recommendedVals = similarityMap.findRepresentativeIsovalues(
      MULTI_MAP_DF_NUMBER, possibleSurfaces);

  vector<RepresentativesInfo> repInfo;


  printResults(recommendedVals, isovalues, fname, repInfo, fields);


  Report::end(total);
  std::ofstream outfile;
  outfile.open(config->outputFolder + "/runtimes.csv", std::ios_base::app);
  outfile << config->variableName << fname << "," << Report::getTotals() << endl;
  outfile.close();
  saveRunInfo(repInfo, mapSize, Report::getRuntime(total));
  cout << endl;
  cout << endl;
  Report::printTotals(true);

  int last = -1;
  float distances[NUM_DISTANCES];
  free(isovalues);
  delete(recommendedVals);
  for(DistanceField& df : fields)
  {
    if(df.isDistancePtrSet())
    {
      free(df.getDistancePtr());
    }
  }
  isosurface->freeImg();
  MemUtils::checkmem("map end");

}




float* Single::compute_isovalues(Settings *settings, float *data, int dataSize, bool uniform)
{
  int histSize = settings->numberOfDistanceFields;

  float min = isosurface->realMinV;
  float max = isosurface->realMaxV;

  float* isovalues = (float *) malloc(histSize * sizeof(float));
  if (uniform)
  {
    int uniformHistSize = 65536;
    //generate the uniform histogram
    //map it back
    //uniform histogram has 64k values in the range from 0-255

    int *hist = (int *) malloc(uniformHistSize * sizeof(int));
    memset(hist, 0, uniformHistSize * sizeof(int));

    get_uniform_histogram(hist, uniformHistSize, data, dataSize, *settings, histSize);

    int idx = 0;
    int lastIdx = 0;
    float step = (max - min) / uniformHistSize;
    for (int i = 0; i < histSize; ++i)
    {
      while (idx < uniformHistSize && hist[idx] <= i)
      {
        ++idx;
      }

      int range = idx - lastIdx;
      int index = lastIdx + range / 2;
      isovalues[i] = min +  step * index;
      lastIdx = idx;
    }

    free(hist);
  } else
  {
    float step = (float) ((max-min)/settings->numberOfDistanceFields);
    for (int i = 0; i < settings->numberOfDistanceFields; ++i)
    {
      isovalues[i] = min + (float) (step * i);
    }

  }
  return isovalues;
}


void Single::get_uniform_histogram(int* hist, int UNIFORM_HISTOGRAM_BIN_COUNT, float* data, float dataSize, Settings settings, int histSize)
{


  int i, index;
  float gmin, gmax, oneovergrange;
  double sum, acc, threshold;



  gmin = isosurface->realMinV, gmax = isosurface->realMaxV;
  oneovergrange = 1.0 / (gmax - gmin);


  for (i = 0; i < dataSize; i++)
  {
    index = int(UNIFORM_HISTOGRAM_BIN_COUNT * (data[i] - gmin) * oneovergrange);

    if (index > UNIFORM_HISTOGRAM_BIN_COUNT - 1)
    { index = UNIFORM_HISTOGRAM_BIN_COUNT - 1; }
    if (index < 0)
    { index = 0; }
    hist[index] += 1.0;
  }


  sum = 0.0;
  // normalize
  for (i = 0; i < UNIFORM_HISTOGRAM_BIN_COUNT; i++)
  {
    sum += hist[i];
  }
  //printfn("sum = %e\n", sum);

  acc = 0;
  index = 0;
  threshold = sum / double(histSize);
  //printfn("threshold = %e\n", threshold);

  for (i = 0; i < histSize; i++)
  {
    while (((acc == 0.0) || ((acc + hist[index]) < threshold)) &&
           (index < UNIFORM_HISTOGRAM_BIN_COUNT - 1))
    {
      acc += hist[index];
      hist[index] = i;
      index++;
    }
    sum -= acc;
    acc = 0.0;
    if (i < histSize - 1)
    {
      threshold = sum / double(histSize - (i + 1));
    }
  }

}

void Single::init()
{
  createFolder();
}

void Single::createFolder()
{
  string approx = settings->approximation ? "-approx" : "";
  string scale = settings->approximation ? "-scale_" + to_string(settings->scale) : "";
  string approxstr = approx + scale;
  string appendix = to_string(settings->similarityMapSize) + "-" +  approxstr;
  outputFolder = settings->outputFolder + "/";

  string varFolder = outputFolder + config->variableName + "-" + to_string(config->timestep);

  if(ISO_OUT)
  {
    isoSurfaceFolder = varFolder + "/isosurfaces-" +  appendix + "/";
    boost::filesystem::create_directories(isoSurfaceFolder);
  }
  if(SIM_PRINT)
  {
    simMapFolder = varFolder + "/similarityMap-" +  appendix + "/";
    boost::filesystem::create_directories(simMapFolder);
  }
  if (DF_OUT)
  {
    distanceFieldFolder = varFolder + DF_FOLDER;
    boost::filesystem::create_directories(distanceFieldFolder);
  }
  if (FILE_LOGGING) {
    logFolder = varFolder + "/log";
    boost::filesystem::create_directories(logFolder);
    logFile = logFolder + "/" + settings->fileName + "-" + appendix + ".log";
  }
}


void Single::printIsosurface(string fileName)
{

  FloatBinaryWriter floatBinaryWriter(fileName);

  unsigned char* data;
  long len = isosurface->printBinary(data);
  cout << "writing to file " << fileName << endl;
  floatBinaryWriter.writeFile(fileName, data, len);
  delete[] data;

}

SimilarityMap Single::calculateHistograms(vector<DistanceField> fields)
{
  vector<vector<JointHistogram> > histograms;
  int histogramSize = settings->histogramSize;

  string similarityMapCalc = "similarity map";
  Report::begin(similarityMapCalc);
  SimilarityMap similarityMap = calculate_histogram_W(fields, (settings->dimX/config->dfDownscale) * (settings->dimY/config->dfDownscale) * (settings->dimZ/config->dfDownscale), histogramSize, fields.size(), false);
  Report::end(similarityMapCalc);

  return similarityMap;
}

void Single::printSimilarityMap(const int mapSize, SimilarityMap* similarityMap, vector<int>& possibleValues)
{
  string app = settings->approximation ? "-approx" : "";
  string s = stringPrintf("%ssimilarityMap-%d%s.bmp", simMapFolder.c_str(), mapSize, app.c_str());
  if (SIM_PRINT)
  {
    if(RESULTS_OUT)
    {
      similarityMap->save(simMapFolder+"map.dat");
    }
  }
}



void Single::printResults(unordered_map<int, SimilarityMap::RepInfo> *recommendedVals, float *isovalues, string fname,
                          vector<RepresentativesInfo> &repInfo, vector<DistanceField> &fields)
{
  if (ISO_OUT || DF_OUT)
  {
    int c = 0;
    int d = 0;
    unordered_map<int,int> order = unordered_map<int,int>();
    vector<int> vals = vector<int>();
    for (auto kv : (*recommendedVals))    {
      vals.push_back(kv.second.id);//todo check this

    }

    sort(vals.begin(), vals.end());
    for (int i : vals)
    {
      order.insert(pair<int, int>(i, c++));
      printf("id %d, # %d\n", i, c);
    }
    c = 0;
    d = 0;


    for (auto kv : (*recommendedVals))
    {
      int fieldId = kv.second.id;

      float curIsovalue = isovalues[fieldId];
      if (ISO_OUT)
      {
        string calc = "calculating representative isosurface nr " + to_string(fieldId) + "/" + to_string(settings->numberOfDistanceFields-1);
        cout << calc << endl;

        if(OUTPUT)
        {
          isosurface->calculateSurfacePoints(settings->approximation, curIsovalue, NULL);
        }
        if (!OUTPUT || !isosurface->isEmpty())
        {
          string sx = stringPrintf("%s%s-%04d-%0.2f-%0.2f.dat", isoSurfaceFolder.c_str(), fname.c_str(), kv.first, curIsovalue, kv.second.isovalue);
          if(OUTPUT)
          {
            printIsosurface(sx);
          }
          RepresentativesInfo representativesInfo = RepresentativesInfo(fieldId, curIsovalue, d++, kv.second.priority, sx, kv.second.mapId);
          repInfo.push_back(representativesInfo);
        }
        isosurface->clear();

      }
      if (DF_OUT)
      {
        cout << "saving distanceFields" << endl;


        int id = c++;
        if(INPUT_ORDER)
        {
          id = order.find(kv.second.id)->second;
        }
        DistanceField* field;
        for (DistanceField& df: fields)
        {
          if (df.id == fieldId)
          {
            field = &df;
          }
        }


        string s = stringPrintf("%sdistancefield-%d.df", distanceFieldFolder.c_str(), id);
        if(RESULTS_OUT)
        {
          field->writeToFile(s);
        }
      }
    }
  }
}


void Single::calculateDistanceField(Isosurface* isosurface, Settings* settings, long length, int3* points, vector<DistanceField>* fields)
{
  bool approx = settings->approximation;
  int dimX = settings->dimX;
  int dimY = settings->dimY;
  int dimZ = settings->dimZ;


  //calculate distance field
  string distanceFieldCalc = "distance field calcuation";
  if(!approx)
  {
    printf("points length: %lu \n", isosurface->points.size());
    fields->push_back(
        DistanceField(isosurface->points, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ));
  } else
  {
    fields->push_back(
        DistanceField(points, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ,
                      length));


  }
  Report::begin(distanceFieldCalc);
  fields->back().calculateDistanceField(false, approx, settings->numberOfSamples);
  Report::end(distanceFieldCalc);

}

void Single::OutputIsosurfaceAndDistancefield(string fname, int index, float curIsovalue, PrintInformation printInfo,
                                              DistanceField *field)
{
  if (ISO_OUT && OUTPUT)
  {
    string s = stringPrintf("%s%s-%04d-%0.2f.obj", isoSurfaceFolder.c_str(),fname.c_str(), index, curIsovalue);
    printIsosurface(s);
  }

  if (DF_OUT && OUTPUT)
  {
    stringstream dfn;
    dfn << distanceFieldFolder << fname << "-df" << setfill('0') << setw(printInfo.leadingZeros) << index << "-" << curIsovalue
        << ".dst";
    field->writeToFile(dfn.str());
  }
}


void Single::saveRunInfo(vector<RepresentativesInfo> repInfo, int mapSize, double runtime)
{
  string jsonFile = outputFolder + config->jsonFile;

  RunInformation runInformation;

  if (( access( jsonFile.c_str(), F_OK ) != -1 ))
  {
    cout << jsonFile << endl;
	runInformation = json::JSONHandler::loadJSON<RunInformation>(jsonFile);
  } else
  {
    std::vector<int> dimensions;
    dimensions.push_back(settings->dimX);
    dimensions.push_back(settings->dimY);
    dimensions.push_back(settings->dimZ);
    runInformation = RunInformation(dimensions);
    runInformation.dfDownscale = config->dfDownscale;
  }
  

  int ts = config->timestep;
  string name = config->variableName;

  runInformation.addVolumeInformation(name, ts, mapSize, repInfo, runtime);

  runInformation.addSimilarityMap(name, name, ts, ts, simMapFolder+"map.dat");

  json::JSONHandler::saveJSON(runInformation, jsonFile);

}

Single::Single(Isosurface *isosurface, Settings *settings, SingleConfig *config)
{
  this->settings = settings;
  this->config = config;
  this->isosurface = isosurface;
  marchingCubes = new MarchingCubes(isosurface);
  init();
}

void Single::clean()
{
  delete(marchingCubes);
}
