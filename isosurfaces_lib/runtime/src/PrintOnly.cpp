//
// Created by mimre on 1/20/17.
//

#include <iostream>
#include "runtime/hdr/PrintOnly.h"
#include "DAO/hdr/RepresentativesInfo.h"
#include "runtime/hdr/globals.h"
#include "DAO/hdr/RunInformation.h"
#include "types/hdr/Isosurface.h"
#include "filehandler/hdr/FileWriter.h"
#include "filehandler/hdr/FloatBinaryWriter.h"
#include "types/hdr/VariableNameAnalyzer.h"

using namespace std;
using namespace filehandler;

void PrintOnly::run()
{

  int dimX = runInformation.dimensions[0];
  int dimY = runInformation.dimensions[1];
  int dimZ = runInformation.dimensions[2];

  Isosurface isosurface(dimX, dimY, dimZ, -1);

  FloatBinaryWriter floatBinaryWriter("");


  for (VolumeInformation vI : runInformation.volumes)
  {
    string curVar = runInformation.variables[vI.variable];
    string filename = stringPrintf("/scratch/MyData/COMBUSTION_480_720_120/jet_hr/jet_hr_0035");

	  cout << filename << endl;
    isosurface.loadFile(filename, false);//false means it's read as float
    for (RepresentativesInfo rI: vI.representatives)
    {
      
	    float curIsovalue = rI.isovalue;
      int curId = rI.valueId;


      string calc = "calculating representative isosurface nr " + to_string(curId) + "/" +
                    to_string(vI.numIsovalues);
      cout << calc << endl;

      isosurface.calculateSurfacePoints(false, curIsovalue, NULL);

      if (!isosurface.isEmpty())
      {
        unsigned char* data;
        long len = isosurface.printBinary(data);
        cout << "writing to file " << rI.filename << endl;
        floatBinaryWriter.writeFile(rI.filename, data, len);
        free(data);

      }
      isosurface.clear();
      isosurface.freeImg();

    }
  }
}

PrintOnly::PrintOnly(const RunInformation &runInformation) : runInformation(runInformation)
{}
