//
// Created by mimre on 7/26/16.
//

#ifndef GLOBALS_H
#define GLOBALS_H


#define APPROXIMATION false
#define SINGLE true
#define HIST_SIZE 128
#define MAP_SIZE 256
#define SCALE 1
#define HIST_EQUALIZATION true
#define NUM_SAMPLES 1500

#define RESULTS_OUT true
#define RECOMMENDATIONS true

#define OUTPUT false
#define DEBUG false
#define ISO_OUT true
#define FILE_LOGGING false
#define PRINTING false
#define SIM_PRINT true
#define DF_OUT true

#define COLOR true

#define INPUT_DOWNSCALE 1

#define MULTI_MAP_DF_NUMBER 16

#define INPUT_ORDER true

#define WTF printf("WTF\n"); std::raise(SIGINT)




#include <string>
#include <stdarg.h>
#include <csignal>


#include "utils/hdr/disablePrint.h"

//const std::string INPUT_ROOT = "/home/mimre/workspace/isosurfaces/";
const std::string INPUT_ROOT = "/scratch/MyData/";
//const std::string DATAFOLDER = INPUT_ROOT+"/data/";
//const std::string DATAFOLDER = INPUT_ROOT + "exploration-test";
//const std::string DATAFOLDER = INPUT_ROOT + "/comb";
const std::string DATAFOLDER = INPUT_ROOT + "/COMBUSTION_480_720_120/jet_Y_OH/";
//const std::string DATAFOLDER = INPUT_ROOT + "/CLIMATE-NEW/DATA-SCIVIS-CLIMATE/salt/";
const std::string OUTPUT_ROOT = "/scratch/isosurfaces/";
//const std::string OUTPUTFOLDER = OUTPUT_ROOT+"/output/paper/";
const std::string OUTPUTFOLDER = OUTPUT_ROOT+"/combustion/";


const std::string DATA_FILE_EXTENSION = ".dat";
const std::string HEADER_FILE_EXTENSION = ".hdr";
const std::string DF_FOLDER = "/distancefields/";
const std::string DF_EXT = ".df";


void printfn (std::string msg, ...);
std::string stringPrintf(std::string msg, ...);
#endif // GLOBALS_H

