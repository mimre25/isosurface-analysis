//
// Created by mimre on 8/2/16.
//

#ifndef FLOATBINARYWRITER_H
#define FLOATBINARYWRITER_H

#include "FileWriter.h"

namespace filehandler
{
  class FloatBinaryWriter : FileWriter
  {
  public:
    FloatBinaryWriter(const string &fileName);

    void writeFile(string fileName, vector <vector<vector < float>>> data);
    void writeFile(string fileName, float* data, long len);
    void writeFile(string fileName, unsigned char* data, long len);
  };
}

#endif //FLOATBINARYWRITER_H
