//
// Created by mimre on 8/2/16.
//

#ifndef BINARYFLOATREADER_H
#define BINARYFLOATREADER_H

#include "FileReader.h"

namespace filehandler
{

  class BinaryFloatReader : FileReader
  {
  public:
    BinaryFloatReader();

    BinaryFloatReader(const string &fileName);

    void read(int numOfDimesions, int dimensions[], vector< vector < vector<float> > >& resultVector);

    void read(int dimX, int dimY, int dimZ, vector< vector < vector<float> > >& resultVector);

    void readBytes(long size, float* output);

    void readBytes(int dimX, int dimY, int dimZ, vector< vector < vector<float> > >& resultVector);
  };
}

#endif //BINARYFLOATREADER_H
