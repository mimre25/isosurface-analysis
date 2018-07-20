//
// Created by mimre on 1/23/17.
//

#include <utils/hdr/StringUtils.h>
#include <utils/hdr/FileUtils.h>
#include "runtime/hdr/Scripting.h"
#include "types/hdr/Settings.h"
#include "runtime/hdr/globals.h"
#include "types/hdr/Isosurface.h"
#include "runtime/hdr/Single.h"
#include "utils/hdr/Report.h"
#include "utils/hdr/MemUtils.h"
#include "DAO/hdr/RunInformation.h"
#include "json/hdr/JSONHandler.h"
#include "utils/hdr/VectorUtils.h"
#include "runtime/hdr/DistanceEqualization.h"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>


void Scripting::run()
{
  Settings* settings;
  Isosurface* isosurface;
  Single* single;
  DistanceEqualization* distanceEqualization;
  SingleConfig* singleConfig = new SingleConfig();
  float* isovalues = nullptr;
  int numIsovalues = NUM_DISTANCES + 1;
  if(TIMESTEP_KEEPING)
  {
    isovalues = (float *) malloc(numIsovalues* sizeof(*isovalues));
  }
  string version = SURFACE_TO_SURFACE_DISTANCE ? "s2s" : "mi";
  string tsKeep = TIMESTEP_KEEPING ? "tsKeep" : "";
  string keepBestSol = KEEP_BEST_SOLUTION ? "keepBest" : "";
  string skipEstimation = NO_ESTIMATION_STAGE ? "skipE" : "";
  string iterationSettings = "MIN_ITR-" + to_string(MIN_ITERATIONS) + "-MAX_ITR-" + to_string(MAX_ITERATIONS) + "-STEPS_P_ITR-" + to_string(BIN_SEARCH_STEP_ITR) + "-ITR-" + to_string(BIN_SEARCH_MAX_ITR);
  string appendix = version + "-" + tsKeep + skipEstimation + "-" + keepBestSol + "-" + iterationSettings;


  for(DataInfo dataInfo : input.entries)
  {

#if !SINGLE
    dataInfo.outputFolder = dataInfo.outputFolder + "-" + to_string(DAMPENING_FACTOR) + "/" + appendix + "/";
#endif
    for (unsigned int j = 0; j < dataInfo.variables.size(); ++j)
    {
      if (TIMESTEP_KEEPING)
      {
        for (int i = 0; i < numIsovalues; ++i)
        {
          isovalues[i] = 0;
        }
      }
      for (unsigned int i = 0; i < dataInfo.timesteps.size(); ++i)
      {
      int curTimestep = dataInfo.timesteps[i];
        MemUtils::checkmem("Before Setup",true);

        Report::clear();
        VariableInfo variableInfo = dataInfo.variables[j];
        string curVar = variableInfo.name;
	      string varFolder = variableInfo.variableFolder;



        string fileName = stringPrintf(dataInfo.fileFormatString.c_str(), dataInfo.inputFolder.c_str(), varFolder.c_str(), curVar.c_str(), curTimestep);
        printf("%s\n", fileName.c_str());

	if ( !boost::filesystem::exists( fileName + ".dat" ) )
{
	cout << fileName <<".dat" << endl;
  continue;

}

#if SINGLE
        settings = new Settings(HIST_SIZE,
                          MAP_SIZE,
                          MAP_SIZE,
                          dataInfo.outputFolder,
                          DATAFOLDER,
                          fileName,
                          dataInfo.dimensions[0],
                          dataInfo.dimensions[1],
                          dataInfo.dimensions[2],
                          variableInfo.minValue,
                          variableInfo.maxValue,
                          SCALE,
                          HIST_EQUALIZATION, /*false means sorting, true means the histogram equalization*/
                          true,
                          NUM_SAMPLES,
                          APPROXIMATION, /*true if you want approximation*/
                          dataInfo.dfDownscale);//TODO merge the branches
#else



        settings = new Settings(HIST_SIZE,
                                numIsovalues,
                                numIsovalues,
                                dataInfo.outputFolder,
                                DATAFOLDER,
                                fileName,
                                dataInfo.dimensions[0],
                                dataInfo.dimensions[1],
                                dataInfo.dimensions[2],
                                variableInfo.minValue,
                                variableInfo.maxValue,
                                SCALE,
                                true,
                                true,
                                NUM_SAMPLES,
                                true,
                                dataInfo.dfDownscale);
#endif



        isosurface = new Isosurface(settings->dimX,
                                    settings->dimY,
                                    settings->dimZ,
                                    -1,
                                    settings->minValue,
                                    settings->maxValue);

        isosurface->loadFile(fileName, false);//todo check settings for type

        MemUtils::checkmem("after setup",true);

        singleConfig->dimensions = dataInfo.dimensions;
        singleConfig->fileName = fileName;
        singleConfig->jsonFile = dataInfo.jsonFile;
        singleConfig->maxValue = variableInfo.maxValue;
        singleConfig->minValue = variableInfo.minValue;
        singleConfig->outputFolder = dataInfo.outputFolder;
        singleConfig->timestep = curTimestep;
        singleConfig->variableName = curVar;
        singleConfig->dfDownscale = dataInfo.dfDownscale;

#if SINGLE
        single = new Single(isosurface, settings, singleConfig);
        MemUtils::checkmem("Before run",true);
        single->run();
        MemUtils::checkmem("after run",true);
        delete(settings);
        delete(isosurface);
        delete(single);
#else
        distanceEqualization = new DistanceEqualization(isosurface, settings, singleConfig);
#ifdef DISABLE_PRINTF
        std::cout.setstate(std::ios_base::failbit);
#endif
        MemUtils::checkmem("before run", true);
        Report::begin("total");
        Report::begin(DistanceEqualization::FOUND_BEST);
        if (CRC)
        {
          try
          {
            distanceEqualization->run(isovalues, false, NO_ESTIMATION_STAGE && i>0);
          } catch (...)
          {
            cout << "error in current run: " + curVar + " " + to_string(curTimestep) << endl;
            distanceEqualization->deleteFolder();
          }
        } else
        {
          distanceEqualization->run(isovalues, false, NO_ESTIMATION_STAGE && i > 0);
        }
        Report::end("total");
#ifdef DISABLE_PRINTF
        std::cout.clear();
#endif
        if (TIMESTEP_KEEPING)
        {
          if(i % 2 == 1)
          {
            for (int k = 0; k < numIsovalues; ++k)
            {
              isovalues[k] = 0.0;
            }
          } else
          {
            for (int k = 0; k < numIsovalues; ++k)
            {
              isovalues[k] = distanceEqualization->bestIsovalues[k];
              printf("isovalue %d: %f\n", k, isovalues[k]);
            }
          }
        }
        else
        {
          isovalues = nullptr;
        }

        distanceEqualization->clean();

        MemUtils::checkmem("after run", true);


        std::ofstream outfile;

        string firstLine = "";
        string fileN = singleConfig->outputFolder + version + "-" + "runtimes.csv";
        if (!FileUtils::fileExists(fileN))
        {
          firstLine = "Runname," + Report::getTotalsNames() +
              "itr est," + "itr ref," +  "err init," + "err est," + "err best,"
              "\n";
        }
        vector<string> fileparts = utils::StringUtils::split(fileName, '/');

        outfile.open(fileN, std::ios_base::app);
        outfile << firstLine;
        outfile << curVar << "-" << curTimestep << "-" << DAMPENING_FACTOR << "," << Report::getTotals()
                << distanceEqualization->iterationInEstimationStage << "," << distanceEqualization->bestSolutionIteration << "," << distanceEqualization->errorAfterInit << "," << distanceEqualization->errorAfterEstimation << "," << distanceEqualization->errorAfterRefinement
                << endl;
        outfile.close();
        delete(settings);
        delete(isosurface);
        delete(distanceEqualization);

#endif

      }

      MemUtils::checkmem("after run2", true);
    }
  }
  free(isovalues);
  delete(singleConfig);
}

Scripting::Scripting(const ScriptingInput &input) : input(input)
{}
