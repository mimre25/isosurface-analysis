#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <vector_functions.h>
#include <cmath>
#include "isosurfaces_cuda/hdr/common.h"
#include "isosurfaces_cuda/hdr/helper_cuda.h"
#include <unistd.h>
#include "isosurfaces_cuda/hdr/common.h"

using namespace std;


#ifdef DISABLE_PRINTF
#define printf(fmt, ...) (0)
#endif


#define MAX_PER_RUN 1000000


__device__ __host__ float calculateMutualInformation(unsigned int* hist, const int sizeX, const int sizeY, const float numValues, unsigned int* colSums, unsigned int* rowSums)
{


  float hX = 0;
  float hY = 0;
  float hXY = 0;

  for(int i = 0; i < sizeX; ++i)
  {
    for (int j = 0; j < sizeY; ++j)
    {
      if (hist[i * sizeY + j] > 0)
      {
        float pxy = hist[i * sizeY + j];
        hXY -= pxy * logf(pxy);//__logf(x)	For x in [0.5, 2], the maximum absolute error is 2-21.41, otherwise, the maximum ulp error is 3.
      }
    }
    if (colSums[i] > 0)
    {
      float px = colSums[i];
      hX -= px * logf(px);
    }
    if (rowSums[i] > 0)
    {
      float py = rowSums[i];
      hY -= py * logf(py);    }


  }
  hXY = hXY/numValues+logf(numValues);
  hX = hX/numValues+logf(numValues);
  hY = hY/numValues+logf(numValues);
  float iXY = hX + hY - hXY;
  float val = 2 * iXY/(hX + hY);
  return val != val ? 0.0f : val;
}


__device__ __host__ int findBucket_d(const float val, const float minValue, const float step, const int histogramSize, int id1, int id2)
{
  if (step == 0)
  {
    return 0;
  }
  int bucket = (int) ((val - minValue) / step);
  if(bucket < 0)
  {
    printf("%d, %d: bucket: %d, %0.2f, %0.2f, %0.2f\n", id1, id2, bucket, val, minValue, step);
  }
  return bucket >= histogramSize ? histogramSize - 1 : bucket;
}

struct runInfo
{
  unsigned long tasksPerRun;
  unsigned long lastRunMemBytes;
  unsigned long histMemMax;
  unsigned long offsetMax;
  unsigned long memoryPerRun;
  bool odd;
  unsigned long maxSize;
  unsigned long fieldMaxMemory;
  bool fieldsLastRun;
  unsigned long fieldsLastRunBytes;
  bool multi;
};

struct dataInfo
{
  const int fieldSize;
  const int histogramSize;
  const int numFields;
  const int similarityMapSize;
};

__global__ void calculate_histogram_g(float *distanceFields, float *minValues, float *maxValues,
                                      unsigned int *jointHist, unsigned int *colSums, unsigned int *rowSums, int2 *tasks, const unsigned long offset,
                                      const unsigned long fieldOffset, float* simMap, runInfo rInfo, dataInfo dInfo)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool last = false;
  if (rInfo.odd && offset + 1 == rInfo.offsetMax)
  {
    last = true; ///this is necessary to see if we have a single run as last run due to memory limitations (and uneven numbers)
  }
  if (idx < rInfo.tasksPerRun && (!last || idx < 1))
  {
    const int fieldSize = dInfo.fieldSize;
    const int histogramSize = dInfo.histogramSize;
    const int numFields = dInfo.numFields;
    unsigned long maxSize = rInfo.maxSize;
    int2 task = tasks[idx + offset * rInfo.tasksPerRun];

    ///get data
    int id1 = task.x;
    int id2 = task.y;
    unsigned long field1Id = static_cast<unsigned long>(id1 * MAX_PER_RUN);
    unsigned long field2Id = static_cast<unsigned long>(id2 * MAX_PER_RUN);
    float *f1 = &distanceFields[field1Id];
    float *f2 = &distanceFields[field2Id];

    ///preparation for bucket selection
    float min = minValues[id1] < minValues[id2] ? minValues[id1] : minValues[id2];
    float max = maxValues[id1] > maxValues[id2] ? maxValues[id1] : maxValues[id2];
    float step = (max - min) / histogramSize;

    ///index calculations
    id2 = rInfo.multi ? id2 - numFields : id2;
    int idx1 = id1 * numFields * histogramSize * histogramSize;
    int idx2 = id2 * histogramSize * histogramSize;
    int histIndex = idx1 + idx2;
    int colRowIndex = id1 * numFields * histogramSize + id2 * histogramSize;



    unsigned long j;


    unsigned long offs = rInfo.fieldsLastRun ? rInfo.fieldsLastRunBytes : MAX_PER_RUN;


    for (j = 0; j < offs && j < fieldSize; ++j)
    {
      int row = findBucket_d(f1[j], min, step, histogramSize, id1, id2);
      int column = findBucket_d(f2[j], min, step, histogramSize, id1, id2);

      ++jointHist[(histIndex + row * histogramSize + column) % maxSize];
      ++colSums[colRowIndex + column];
      ++rowSums[colRowIndex + row];
    }
    if (rInfo.fieldsLastRun && j == offs)
    {
      simMap[id1 * dInfo.similarityMapSize + id2] =
          calculateMutualInformation(&jointHist[histIndex % maxSize], histogramSize, histogramSize, fieldSize, &colSums[colRowIndex], &rowSums[colRowIndex]);
    }

  }
}


__host__ void calculate_histogram_CPU(float *distanceFields, float *minValues, float *maxValues,
                                      unsigned int *jointHist, unsigned int *colSums, unsigned int *rowSums, int2 *tasks, const int offset,
                                      const int fieldOffset, float* simMap, runInfo rInfo, dataInfo dInfo)
{
  for (int idx = 0; idx < rInfo.tasksPerRun; ++idx)
  {

    bool last = false;
    if (rInfo.odd && offset + 1 == rInfo.offsetMax)
    {
      last = true; ///this is necessary to see if we have a single run as last run due to memory limitations (and uneven numbers)
    }
    if (idx < rInfo.tasksPerRun && (!last || idx < 1))
    {
      const int fieldSize = dInfo.fieldSize;
      const int histogramSize = dInfo.histogramSize;
      const int numFields = dInfo.numFields;
      unsigned long maxSize = rInfo.maxSize;
      int2 task = tasks[idx + offset * rInfo.tasksPerRun];

      ///get data
      int id1 = task.x;
      int id2 = task.y;
      int field1Id = id1 * MAX_PER_RUN;
      int field2Id = id2 * MAX_PER_RUN;
      float *f1 = &distanceFields[field1Id];
      float *f2 = &distanceFields[field2Id];

      ///preparation for bucket selection
      float min = minValues[id1] < minValues[id2] ? minValues[id1] : minValues[id2];
      float max = maxValues[id1] > maxValues[id2] ? maxValues[id1] : maxValues[id2];
      float step = (max - min) / histogramSize;

      ///index calculations
      id2 = rInfo.multi ? id2 - numFields : id2;
      int idx1 = id1 * numFields * histogramSize * histogramSize;
      int idx2 = id2 * histogramSize * histogramSize;
      int histIndex = idx1 + idx2;
      int colRowIndex = id1 * numFields * histogramSize + id2 * histogramSize;


      int j;

      unsigned long offs = rInfo.fieldsLastRun ? rInfo.fieldsLastRunBytes : MAX_PER_RUN;


      for (j = 0; j < offs && j < fieldSize; ++j)
      {
        int row = findBucket_d(f1[j], min, step, histogramSize, id1,id2);
        int column = findBucket_d(f2[j], min, step, histogramSize, id1, id2);
        ++jointHist[(histIndex + row * histogramSize + column) % maxSize];
        ++colSums[colRowIndex + column];
        ++rowSums[colRowIndex + row];
      }
      if (rInfo.fieldsLastRun && j == offs)
      {
        simMap[id1 * dInfo.similarityMapSize + id2] =
            calculateMutualInformation(&jointHist[histIndex % maxSize], histogramSize, histogramSize, fieldSize,
                                       &colSums[colRowIndex], &rowSums[colRowIndex]);
      }

    }
  }
}


runInfo calculateRunInfo(const int numTasks, const unsigned long HISTOGRAM_MEMORY_SIZE, unsigned long NUM_HISTOGRAMS)
{

  unsigned long offsetMax = 1;
  unsigned long histMemMax = HISTOGRAM_MEMORY_SIZE;
  unsigned long lastRunMemBytes = 0;
  if (HISTOGRAM_MEMORY_SIZE > MALLOC_MAX_BYTES)
  {
    while (++offsetMax * MALLOC_MAX_BYTES < HISTOGRAM_MEMORY_SIZE)
    {}
    histMemMax = HISTOGRAM_MEMORY_SIZE / offsetMax;
  }
  lastRunMemBytes = histMemMax;


  //we know we need 2 runs, to fit everything into the memory
  //now we check how many taskList per run we can do
  unsigned long tasksPerRun = numTasks / offsetMax;


  //check if we have an odd amount of runs
  bool odd = numTasks % 2 == 1 && offsetMax > 1; // if we have to do more than one run it's important to know if we have an odd amount.
  unsigned long memoryPerRun = histMemMax;
  unsigned long MEM_PER_HIST = HISTOGRAM_MEMORY_SIZE / NUM_HISTOGRAMS;
  if (odd)
  {
    memoryPerRun = histMemMax - MEM_PER_HIST / offsetMax;
    ++offsetMax;
    lastRunMemBytes = MEM_PER_HIST;
  }

  runInfo r = {tasksPerRun, lastRunMemBytes, histMemMax, offsetMax, memoryPerRun, odd, MAX_PER_RUN * sizeof(float)};
  return r;
}

extern "C"
__host__ void calculate_histogram_h(float **distanceFields, float *minValues, float *maxValues, const int fieldSize,
                                    const unsigned long numFields, const unsigned long histogramSize,
                                    unsigned int *h_jointHist, unsigned int *h_colSums, unsigned int *h_rowSums,
                                    float* h_simMap, const int simMapSize, const bool multi)
{
///setup
  printf("calculate_histogram_h start\n");

  printf("accolacting memory on gpu\n");


  float *d_distanceFields;
  float *d_minValues;
  float *d_maxValues;
  unsigned int *d_jointHist;
  unsigned int *d_colSums;
  unsigned int *d_rowSums;
  float *d_simMap;
  unsigned long NUM_HISTOGRAMS = (numFields * numFields);
  unsigned long MINMAX_MEMORY_SIZE = numFields * sizeof(*d_minValues) * (multi ? 2 : 1);
  unsigned long HISTOGRAM_MEMORY_SIZE = numFields * numFields * histogramSize * histogramSize * sizeof(*d_jointHist);
  unsigned long SUMS_MEMORY_SIZE = numFields * numFields * histogramSize * sizeof(*d_colSums);
  unsigned long SIMMAP_MEMORY_SIZE = simMapSize * simMapSize * sizeof(*d_simMap);

  checkCudaErrors(cudaMalloc((void **) &d_distanceFields, numFields * MAX_PER_RUN * sizeof(*d_distanceFields) * (multi ? 2 : 1)));
  checkCudaErrors(cudaMalloc((void **) &d_minValues, MINMAX_MEMORY_SIZE));
  checkCudaErrors(cudaMalloc((void **) &d_maxValues, MINMAX_MEMORY_SIZE));
  checkCudaErrors(cudaMalloc((void **) &d_colSums, SUMS_MEMORY_SIZE));
  checkCudaErrors(cudaMalloc((void **) &d_rowSums, SUMS_MEMORY_SIZE));
  checkCudaErrors(cudaMalloc((void **) &d_simMap, SIMMAP_MEMORY_SIZE));

  checkCudaErrors(cudaMemset(d_colSums, 0, SUMS_MEMORY_SIZE));
  checkCudaErrors(cudaMemset(d_rowSums, 0, SUMS_MEMORY_SIZE));

  checkCudaErrors(cudaMemcpy(d_minValues, minValues, MINMAX_MEMORY_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_maxValues, maxValues, MINMAX_MEMORY_SIZE, cudaMemcpyHostToDevice));
///put into setup function

  printf("creating tasks\n");
  size_t taskSize = numFields * numFields * sizeof(int2);
  int2 *taskList = (int2 *) malloc(taskSize);
  size_t c = 0;
  if (multi)
  {
    for (int i = 0; i < numFields; ++i)
    {
      for (int j = 0; j < numFields; ++j)
      {
        taskList[i * numFields + j] = make_int2(i, j+numFields);
        ++c;
      }
    }
  } else
  {
    for (int i = 0; i < numFields; ++i)
    {
      for (int j = 0; j < numFields; ++j)
      {
        taskList[c++] = make_int2(i, j);
      }
    }
  }

  size_t numTasks = c;

  int2 *d_tasks;
  taskSize = numTasks * sizeof(*d_tasks);
  checkCudaErrors(cudaMalloc((void **) &d_tasks, taskSize));
  checkCudaErrors(cudaMemcpy(d_tasks, taskList, taskSize, cudaMemcpyHostToDevice));
  free(taskList);

  ///finding the loop rounds
  printf("getting runInfo\n");
  runInfo rInfo = calculateRunInfo(numTasks, HISTOGRAM_MEMORY_SIZE, NUM_HISTOGRAMS);
  rInfo.multi = multi;
//  int tasksPerRun = rInfo.tasksPerRun;
  unsigned long lastRunMemBytes = rInfo.lastRunMemBytes;
  unsigned long histMemMax = rInfo.histMemMax;
  unsigned long offsetMax = rInfo.offsetMax;
  unsigned long memoryPerRun = rInfo.memoryPerRun;
  bool odd = rInfo.odd;
  rInfo.maxSize = memoryPerRun / sizeof(unsigned int);
  printf("memory per run: %lu\n", memoryPerRun);
  printf("getting data info\n");

  dataInfo dInfo = {fieldSize, histogramSize, numFields, simMapSize};
  checkCudaErrors(cudaMalloc((void **) &d_jointHist, memoryPerRun));



  printf("computing launch params\n");
  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
  // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on input size
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculate_histogram_g, 0, 0);
  // Round up according to array size
  gridSize = (((int)rInfo.tasksPerRun) + blockSize - 1) / blockSize;
  size_t f, t;
  checkCudaErrors(cudaMemGetInfo(&f, &t));



  ///main task

  for (unsigned long offset = 0; offset < offsetMax; ++offset)
  {
    checkCudaErrors(cudaMemset(d_jointHist, 0, memoryPerRun));
    unsigned long fieldsBytes = MAX_PER_RUN * sizeof(float);
    rInfo.fieldsLastRun = false;
    for (unsigned long fieldOffset = 0; fieldOffset < fieldSize; fieldOffset += MAX_PER_RUN)
    {
      if (fieldOffset + MAX_PER_RUN >= fieldSize)
      {
        rInfo.fieldsLastRun = true;
        rInfo.fieldsLastRunBytes = (unsigned long) (fieldSize % MAX_PER_RUN);
        fieldsBytes = rInfo.fieldsLastRunBytes * sizeof(float);
      }
      for (unsigned long i = 0; i < numFields*(multi ? 2 : 1); ++i)
      {
        checkCudaErrors(cudaMemcpy(&d_distanceFields[i*MAX_PER_RUN], &distanceFields[i][fieldOffset], fieldsBytes, cudaMemcpyHostToDevice));
      }
      calculate_histogram_g << < gridSize , blockSize >> >
                                            (d_distanceFields, d_minValues, d_maxValues,
                                                d_jointHist, d_colSums, d_rowSums, d_tasks, offset,
                                                fieldOffset, d_simMap, rInfo, dInfo);
    }
    int ix = (int) (offset * (memoryPerRun / sizeof(int)));

    unsigned long bytesToCopy = offset + 1 == offsetMax ? lastRunMemBytes : memoryPerRun;
//    checkCudaErrors(cudaMemcpy(&h_jointHist[ix], d_jointHist, bytesToCopy, cudaMemcpyDeviceToHost));

  }



  ///cleanup
  checkCudaErrors(cudaFree(d_jointHist));

  checkCudaErrors(cudaMemcpy(h_colSums, d_colSums, SUMS_MEMORY_SIZE, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_rowSums, d_rowSums, SUMS_MEMORY_SIZE, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_simMap, d_simMap, SIMMAP_MEMORY_SIZE, cudaMemcpyDeviceToHost));



  checkCudaErrors(cudaFree(d_distanceFields));
  checkCudaErrors(cudaFree(d_minValues));
  checkCudaErrors(cudaFree(d_maxValues));
  checkCudaErrors(cudaFree(d_colSums));
  checkCudaErrors(cudaFree(d_rowSums));
  checkCudaErrors(cudaFree(d_tasks));
  checkCudaErrors(cudaFree(d_simMap));
  return;
}

