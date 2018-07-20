//
// Created by mimre on 20.06.18.
//

#include <filehandler/hdr/BinaryFloatReader.h>
#include "runtime/hdr/CheckMap.h"

CheckMap::CheckMap(const std::string &fileName) : fileName(fileName)
{}


int CheckMap::run()
{
  printf("reading file %s\n",fileName.c_str());
  filehandler::BinaryFloatReader fr(fileName);
  size_t n = 255;
  size_t size = n*n;
  float* value = static_cast<float *>(malloc(size * sizeof(float)));
  fr.readBytes(size, value);
  printf("finished reading file\n");
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      printf("%f,", value[i*n +j]);
    }
    printf("\n");
  }

}