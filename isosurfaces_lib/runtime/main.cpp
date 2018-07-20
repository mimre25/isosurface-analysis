
#include <boost/program_options.hpp>
#include "isosurfaces_cuda/hdr/funcs.h"
#include <runtime/hdr/CreateMapOnly.h>
#include <runtime/hdr/Stitch.h>
#include <runtime/hdr/Check.h>
#include <runtime/hdr/CheckMap.h>
#include <runtime/hdr/Histogram.h>
#include "types/hdr/MarchingCubes.h"
#include "runtime/hdr/MultiVar.h"
#include "runtime/hdr/Single.h"
#include "json/hdr/JSONHandler.h"
#include "runtime/hdr/PrintOnly.h"
#include "DAO/hdr/ScriptingInput.h"
#include "runtime/hdr/Scripting.h"
#include "utils/hdr/MemUtils.h"
#include "DAO/hdr/SingleConfig.h"
#include "runtime/hdr/DistanceEqualization.h"
#include "runtime/hdr/SurfaceToSurface.h"
#include "types/hdr/MarchingCubes.h"
#include "types/hdr/Settings.h"

using namespace boost::program_options;

const char* HELP = "help";

const char* FILE_NAME = "file name";
const char* DIM_X = "x dimension";
const char* DIM_Y = "y dimension";
const char* DIM_Z = "z dimension";

const char* INPUT_1 = "INPUT_1";
const char* INPUT_2 = "INPUT_2";

const char* JSON_FILE = "JSON_FILE";

int singleModule(int argc, const char *argv[]);
int multiModule(int argc, const char *argv[]);
int printModule(int argc, const char *argv[]);
int scriptingModule(int argc, const char *argv[]);
int distanceModule(int argc, const char *argv[]);
int surfaceToSurfaceModule(int argc, const char *argv[]);
int mapModule(int argc, const char *argv[]);
int stitchModule(int argc, const char *argv[]);
int checkModule(int argc, const char *argv[]);
int checkMapModule(int argc, const char *argv[]);
int histogramModule(int argc, const char *argv[]);



int main(int argc, const char *argv[])
{

  string helpText = "You need to specify a module:\n\tprogram [sign|kf]\n\nUse   program <module> --help   for a description.";

  if (argc < 2)
  {  // Then no module was given
    cout << helpText << endl;
    return -1;
  }

  string module = argv[1];  // This is the first argument. Index 0 holds the executable

  // Copy argv but exclude argv[1}
  int moduleArgC = argc - 1;
  char *moduleArgV[moduleArgC];
  moduleArgV[0] = (char *) argv[0];
  for (int i = 1; i < moduleArgC; ++i)
  {
    moduleArgV[i] = (char *) argv[i + 1];
  }

  if (module == "multi")
  {
    return multiModule(moduleArgC, (const char **) moduleArgV);
  } else if (module == "single")
  {
    return singleModule(moduleArgC, (const char **) moduleArgV);
  } else if (module == "print")
  {
    return printModule(moduleArgC, (const char **) moduleArgV);
  } else if (module == "script")
  {
    return scriptingModule(moduleArgC, (const char**) moduleArgV);
  }  else if (module == "distance")
  {
    return distanceModule(moduleArgC, (const char**) moduleArgV);
  } else if (module == "s2s")
  {
    return surfaceToSurfaceModule(moduleArgC, (const char**) moduleArgV);
  } else if (module == "map")
  {
    return mapModule(moduleArgC, (const char **) moduleArgV);
  } else if (module == "stitch")
  {
    return stitchModule(moduleArgC, (const char **) moduleArgV);
  } else if (module == "check")
  {
    return checkModule(moduleArgC, (const char**) moduleArgV);
  } else if (module == "checkmap")
  {
    return checkMapModule(moduleArgC, (const char**) moduleArgV);
  } else if (module == "histogram")
  {
    return histogramModule(moduleArgC, (const char**) moduleArgV);
  }
  else {  // default
    cout << "Unknown module: " << module << "\n\n" << helpText << endl;
    return -1;
  }
}

int printModule(int argc, const char *argv[])
{
  cout << "print" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, 1);
  desc.add_options()
      (JSON_FILE, value<string>()->required(), "JSON File");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);


  RunInformation runInformation = json::JSONHandler::loadJSON<RunInformation>(vm[JSON_FILE].as<string>());
  PrintOnly po(runInformation);
  po.run();


  return 0;

}

int mapModule(int argc, const char *argv[])
{
  cout << "print" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, 1);
  desc.add_options()
      (JSON_FILE, value<string>()->required(), "JSON File");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);


  RunInformation runInformation = json::JSONHandler::loadJSON<RunInformation>(vm[JSON_FILE].as<string>());
  CreateMapOnly cmo(runInformation);
  cmo.run();


  return 0;

}


int multiModule(int argc, const char *argv[])
{
  cout << "multi" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, 1);
  desc.add_options()
      (JSON_FILE, value<std::string>()->required(), "jsonFile");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);
  MultiVar mv = MultiVar(vm[JSON_FILE].as<string>());
  
cout << "starting multi" << endl;
	mv.run();


  return 0;

}

int singleModule(int argc, const char *argv[])
{
  cout << "single" << endl;
  options_description desc("Allowed options");
  positional_options_description p;
  p.add(FILE_NAME, 1);



  desc.add_options()
      (HELP, "produce help message")
      (FILE_NAME, value<string>()->required(), "name of the config file")
      ;


  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  if (vm.count(HELP)) {
    cout << desc << "\n";
    return 1;
  }
  string fileName = vm[FILE_NAME].as<string>();
  string outputFolder = OUTPUTFOLDER + fileName + "/";

  SingleConfig singleConfig = json::JSONHandler::loadJSON<SingleConfig>(fileName);


  Settings settings(HIST_SIZE,
                    MAP_SIZE,
                    MAP_SIZE,
                    singleConfig.outputFolder,
                    DATAFOLDER,
                    singleConfig.fileName,
                    singleConfig.dimensions[0],
                    singleConfig.dimensions[1],
                    singleConfig.dimensions[2],
                    singleConfig.minValue,
                    singleConfig.maxValue,
                    SCALE,
                    true,
                    true,
                    NUM_SAMPLES,
                    true,
                    singleConfig.dfDownscale);


  Isosurface isosurface(settings.dimX,
                        settings.dimY,
                        settings.dimZ,
                        -1,
                        settings.minValue,
                        settings.maxValue);

  isosurface.loadFile(singleConfig.fileName, false);//todo check settings for type
  Single* single = new Single(&isosurface, &settings, &singleConfig);
  MemUtils::checkmem("before run", true);
  single->run();
  single->clean();
  delete(single);
  MemUtils::checkmem("after run", true);

  return 0;
}

int scriptingModule(int argc, const char* argv[])
{
  cout << "script" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, 1);
  desc.add_options()
      (JSON_FILE, value<string>()->required(), "JSON File");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);


  ScriptingInput scriptingInput = json::JSONHandler::loadJSON<ScriptingInput>(vm[JSON_FILE].as<string>());
  cout << "json file loaded" << endl;
  Scripting sc(scriptingInput);
  sc.run();

  return 0;
}

int distanceModule(int argc, const char *argv[])
{
  cout << "distance" << endl;
  options_description desc("Allowed options");
  positional_options_description p;
  p.add(FILE_NAME, 1);



  desc.add_options()
      (HELP, "produce help message")
      (FILE_NAME, value<string>()->required(), "name of the config file")
      ;


  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  if (vm.count(HELP)) {
    cout << desc << "\n";
    return 1;
  }
  string fileName = vm[FILE_NAME].as<string>();
  string outputFolder = OUTPUTFOLDER + fileName + "/";

  SingleConfig singleConfig = json::JSONHandler::loadJSON<SingleConfig>(fileName);


  Settings settings(NUM_DISTANCES+1,
                    NUM_DISTANCES+1,
                    NUM_DISTANCES+1,
                    singleConfig.outputFolder,
                    DATAFOLDER,
                    singleConfig.fileName,
                    singleConfig.dimensions[0],
                    singleConfig.dimensions[1],
                    singleConfig.dimensions[2],
                    singleConfig.minValue,
                    singleConfig.maxValue,
                    SCALE,
                    true,
                    true,
                    NUM_SAMPLES,
                    true,
                    singleConfig.dfDownscale);


  Isosurface isosurface(settings.dimX,
                        settings.dimY,
                        settings.dimZ,
                        -1,
                        settings.minValue,
                        settings.maxValue);

  isosurface.loadFile(singleConfig.fileName, false);//todo check settings for type
  DistanceEqualization* distanceEqualization = new DistanceEqualization(&isosurface, &settings, &singleConfig);
  MemUtils::checkmem("before run", true);
  distanceEqualization->run(nullptr, false, false);
  distanceEqualization->clean();
  delete(distanceEqualization);
  MemUtils::checkmem("after run", true);

  return 0;
}

int surfaceToSurfaceModule(int argc, const char *argv[])
{
  cout << "surface2surface" << endl;
  options_description desc("Allowed options");
  positional_options_description p;
  p.add(FILE_NAME, 1);



  desc.add_options()
      (HELP, "produce help message")
      (FILE_NAME, value<string>()->required(), "name of the config file")
      ;


  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  if (vm.count(HELP)) {
    cout << desc << "\n";
    return 1;
  }
  string fileName = vm[FILE_NAME].as<string>();
  string outputFolder = OUTPUTFOLDER + fileName + "/";

  SingleConfig singleConfig = json::JSONHandler::loadJSON<SingleConfig>(fileName);


  Settings settings(HIST_SIZE,
                    MAP_SIZE,
                    MAP_SIZE,
                    singleConfig.outputFolder,
                    DATAFOLDER,
                    singleConfig.fileName,
                    singleConfig.dimensions[0],
                    singleConfig.dimensions[1],
                    singleConfig.dimensions[2],
                    singleConfig.minValue,
                    singleConfig.maxValue,
                    SCALE,
                    true,
                    true,
                    NUM_SAMPLES,
                    true,
                    singleConfig.dfDownscale);


  Isosurface isosurface(settings.dimX,
                        settings.dimY,
                        settings.dimZ,
                        -1,
                        settings.minValue,
                        settings.maxValue);

  isosurface.loadFile(singleConfig.fileName, false);//todo check settings for type
  Isosurface isosurface2(settings.dimX,
                        settings.dimY,
                        settings.dimZ,
                        -1,
                        settings.minValue,
                        settings.maxValue);

  isosurface2.loadFile(singleConfig.fileName, false);//todo check settings for type
  SurfaceToSurface* s2s = new SurfaceToSurface(&isosurface, &isosurface2, &settings, &singleConfig);
  MemUtils::checkmem("before run", true);
  s2s->run();
  delete(s2s);
  MemUtils::checkmem("after run", true);

  return 0;
}


/**
 * stitches the files together, using the first one as host file.
 */
int stitchModule(int argc, const char *argv[])
{
  cout << "stitch" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, -1);
  desc.add_options()
      (JSON_FILE, value<vector<string>>(), "JSON Files");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  std::vector<string> files = vm[JSON_FILE].as<std::vector<string>>();
  printf("files: \n");
  for (auto f : files)
  {
    printf("%s\n", f.c_str());
  }

  if(files.size() < 2)
  {
    printf ("Error: at least 2 files needed to stitch");
  } else
  {
    Stitch stitch(files);
    stitch.run();
  }


  return 0;

}

/**
 * checks if all the multimaps are computed
 * @param argc
 * @param argv jsonfile, first ts, last ts
 * @return
 */
int checkModule(int argc, const char *argv[])
{
  cout << "check" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, 1)
      .add(INPUT_1, 1)
      .add(INPUT_2, 1);
  desc.add_options()
      (JSON_FILE, value<string>(), "JSON Files")
      (INPUT_1, value<int>(), "First time step")
      (INPUT_2, value<int>(), "Last time step");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  string file = vm[JSON_FILE].as<string>();
  int ts1 = vm[INPUT_1].as<int>();
  int ts2 = vm[INPUT_2].as<int>();

  if (ts1 > ts2)
  {
    std::swap(ts1, ts2);
  }

  Check check(file, ts1, ts2);
  check.run();



  return 0;

}


/**
 * module to debug similaritymap priority thingy
 * @param argc
 * @param argv
 * @return
 */
int checkMapModule(int argc, const char *argv[])
{
  cout << "CheckMap" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, 1);
  desc.add_options()
      (JSON_FILE, value<string>(), "map file");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  string file = vm[JSON_FILE].as<string>();

  CheckMap checkMap(file);
  checkMap.run();


  return 0;

}


/**
 * computes the joint histogram/similarity map from distancefields stored on disc
 * @param argc
 * @param argv
 * @return
 */
int histogramModule(int argc, const char *argv[])
{
  cout << "histogram" << endl;

  options_description desc("Allowed options");
  positional_options_description p;
  p.add(JSON_FILE, -1);
  p.add(DIM_X, 1);
  p.add(DIM_Y, 1);
  p.add(DIM_Z, 1);
  desc.add_options()
      (JSON_FILE, value<vector<string>>(), "Distance Fields")
      (DIM_X, value<int>(), "dimX")
      (DIM_Y, value<int>(), "dimY")
      (DIM_Z, value<int>(), "dimZ");

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  std::vector<string> files = vm[JSON_FILE].as<std::vector<string>>();
  int dimX = vm[DIM_X].as<int>();
  int dimY = vm[DIM_Y].as<int>();
  int dimZ = vm[DIM_Z].as<int>();
  printf("files: \n");
  for (auto f : files)
  {
    printf("%s\n", f.c_str());
  }

  if(files.size() < 128)
  {
    printf ("Error: at least 128 files needed to stitch");
  } else
  {
    Histogram hist(files, dimX, dimY, dimZ);
    hist.run();
  }


  return 0;

}