//
// Created by mimre on 8/2/16.
//

#include <cstring>
#include "filehandler/hdr/BinaryFloatReader.h"
#include "runtime/hdr/globals.h"
#include <fstream>


namespace filehandler
{


  BinaryFloatReader::BinaryFloatReader(const string &fileName) : FileReader(fileName)
  {
    setReadMode(FileReader::BINARY_READ);
  }

  void BinaryFloatReader::read(int numOfDimesions, int dimensions[], vector< vector < vector<float> > > &resultVector)
  {

    throw("Implement me");
    numOfDimesions++;
    dimensions[0]++;
    resultVector.resize(3);
  }

  void BinaryFloatReader::read(int dimX, int dimY, int dimZ, vector< vector < vector<float> > > &resultVector)
  {
    //read data file
    setFileName(fileName);
    unsigned char *binData = get_file_contents();
    resultVector.resize((unsigned long) dimX);
    for (int x = 0; x < dimX; ++x)
    {
      resultVector[x].resize((unsigned long) dimY);
      for (int y = 0; y < dimY; ++y)
      {
        resultVector[x][y].resize((unsigned long) dimZ);
        for (int z = 0; z < dimZ; ++z)
        {
          int srcIdx = sizeof(float) * (x + y * dimX + z * dimY * dimX);
          memcpy(&resultVector[x][y][z], &binData[srcIdx], sizeof(float));
        }
      }
    }
    free(binData);
  }

  void BinaryFloatReader::readBytes(int dimX, int dimY, int dimZ, vector< vector < vector<float> > > &resultVector)
  {
    //read data file
    setFileName(fileName);
    unsigned char *binData = get_file_contents();
    resultVector.resize((unsigned long) dimX);
    for (int x = 0; x < dimX/INPUT_DOWNSCALE; ++x)
    {
      resultVector[x].resize((unsigned long) dimY);
      for (int y = 0; y < dimY/INPUT_DOWNSCALE; ++y)
      {
        resultVector[x][y].resize((unsigned long) dimZ);
        for (int z = 0; z < dimZ/INPUT_DOWNSCALE; ++z)
        {
          int srcIdx = (x*INPUT_DOWNSCALE + y*INPUT_DOWNSCALE * dimX + z*INPUT_DOWNSCALE * dimY * dimX);
          unsigned char b;
          memcpy(&b, &binData[srcIdx], 1);
          resultVector[x][y][z] = (float) b;
        }
      }
    }
    free(binData);
  }

  BinaryFloatReader::BinaryFloatReader()
  {
    setReadMode(FileReader::BINARY_READ);
  }

  void BinaryFloatReader::readBytes(long size, float *output)
  {
    evaluateFileOpenMode(getReadMode());
    ifstream file(fileName, fileOpenMode);
    long f = file.readsome((char *) output, size * sizeof(float));
    file.close();
  }
}