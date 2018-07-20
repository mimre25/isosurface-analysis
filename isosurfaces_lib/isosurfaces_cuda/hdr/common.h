#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#include <stdio.h>
#include "utils/hdr/disablePrint.h"


const unsigned long MALLOC_MAX_BYTES = 1u<<31;//2GB

int calculateBlockNumber(unsigned long totalSize, int blockSize);

int calculateBlockNumber(int totalSize, int blockSize);

struct Reporter
{
  char* msg;
  cudaEvent_t eventStart;
  cudaEvent_t eventEnd;
  void reportStart(char* msg)
  {
    this->msg = msg;
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventEnd);
    cudaEventRecord(eventStart, 0);
  };
  void reportEnd()
  {
    cudaEventRecord(eventEnd, 0);
    cudaEventSynchronize(eventEnd);
    float miliseconds;
    cudaEventElapsedTime(&miliseconds, eventStart, eventEnd);
    cudaEventDestroy(eventStart);
    cudaEventDestroy(eventEnd);
    printf("CUDA REPORT: %s: %f ms\n", msg, miliseconds);
  };
};

#endif //__CUDA_COMMON_H__