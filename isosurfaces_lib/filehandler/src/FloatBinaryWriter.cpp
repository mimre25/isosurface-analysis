//
// Created by mimre on 8/2/16.
//

#include <fstream>
#include "filehandler/hdr/FloatBinaryWriter.h"
namespace filehandler{

  FloatBinaryWriter::FloatBinaryWriter(const string &fileName) : FileWriter(fileName)
  {
    setWMode(BINARY_WRITE_READ);
  }

  void FloatBinaryWriter::writeFile(string fileName, vector <vector<vector < float >>> data)
  {
    setFileName(fileName);
    evaluateFileOpenMode(getWMode());
    ofstream file(fileName, fileOpenMode);

    for (int i = 0; i < data.size(); ++i)
    {
      for (int j = 0; j < data[i].size(); ++j)
      {
        const unsigned char* buffer = (const unsigned char *) (&data[i][j][0]);
        file.write((const char *) buffer, data[i][j].size()*sizeof(float));
      }
    }

    file.close();

  }

  void FloatBinaryWriter::writeFile(string fileName, float *data, long len)
  {
    setFileName(fileName);
    evaluateFileOpenMode(getWMode());
    ofstream file(fileName, fileOpenMode);
    const unsigned char* buffer = (const unsigned char *) (&data[0]);
    file.write((const char *) buffer, len*sizeof(*data));
    file.close();

  }

  void FloatBinaryWriter::writeFile(string fileName, unsigned char *data, long len)
  {
    setFileName(fileName);
    evaluateFileOpenMode(getWMode());
    ofstream file(fileName, fileOpenMode);
    file.write((const char *) data, len);
    file.close();
  }
}