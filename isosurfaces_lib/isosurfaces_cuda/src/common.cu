#include <driver_types.h>
#include <cuda_runtime.h>
#include "isosurfaces_cuda/hdr/common.h"

int calculateBlockNumber(unsigned long totalSize, int blockSize)
{
  int numberOfBlocks = totalSize/blockSize;
  if (totalSize % blockSize != 0)
  {
    ++numberOfBlocks;
  }
  return numberOfBlocks;
}

int calculateBlockNumber(int totalSize, int blockSize)
{
  return calculateBlockNumber((unsigned long) totalSize, blockSize);
}


