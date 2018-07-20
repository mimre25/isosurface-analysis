//
// Created by mimre on 8/8/16.
//

#include <cublas_v2.h>
#include <limits>
#include <iostream>
#include "types/hdr/JointHistogram.h"

void JointHistogram::calculateHistogram(DistanceField* field1, DistanceField* field2)
{
  int dimX = field1->getDimX();
  int dimY = field1->getDimY();
  int dimZ = field1->getDimZ();
  if (dimX != field2->getDimX() || dimY != field2->getDimY() || dimZ != field2->getDimZ())
  {
    throw ("Dimension mismatch");
  }

  vector< vector< vector<float> > > distances1 = field1->getDistances();
  vector< vector< vector<float> > > distances2 = field2->getDistances();
  min = 0;
  max = numeric_limits<int>::min();
  for (int x = 0; x < dimX; ++x)
  {
    for (int y = 0; y < dimY; ++y)
    {
      for (int z = 0; z < dimZ; ++z)
      {
        float val1 = distances1[x][y][z];
        float val2 = distances2[x][y][z];

        int tmp = ++histogram[findBucket(val1)][findBucket(val2)];
        if (tmp > max)
        {
          max = tmp;
        }
      }
    }
  }

  colSums.resize((unsigned long) size);
  rowSums.resize((unsigned long) size);
  numValues = 0;
  for (int i = 0; i < size; ++i)
  {
    colSums[i] = 0;
    rowSums[i] = 0;
    for (int j = 0; j < size; ++j)
    {
      colSums[i] += histogram[j][i];
      rowSums[i] += histogram[i][j];
      numValues += histogram[i][j];
    }
  }

}

int JointHistogram::getNumValues() const
{
  return numValues;
}

const vector<int> &JointHistogram::getRowSums() const
{
  return rowSums;
}

const vector<int> &JointHistogram::getColSums() const
{
  return colSums;
}

int JointHistogram::getMin() const
{
  return min;
}

int JointHistogram::getMax() const
{
  return max;
}

int JointHistogram::findBucket(float val)
{
  if (step == 0)
  {
    return 0;
  }
  int bucket = (int) ((val - minValue) / step);
  return bucket >= size ? size - 1 : bucket;
}


void JointHistogram::print()
{
  printf("intervals:\n");
  for (float i = minValue; i < maxValue; i+=step)
  {
    printf("%f,",i);
  }
  printf("\n");

  for (int i = 0; i < size; ++i)
  {
    printf("{");
    for (int j = 0; j < size; ++j)
    {
      printf("%d,", histogram[i][j]);
    }
    printf("} %d\n", rowSums[i]);
  }
  for (int k = 0; k < size; ++k)
  {
    printf("%d,", colSums[k]);
  }
  printf("\n");
}

JointHistogram::JointHistogram(int size, int min, int max) : size(size), min(min), max(max)
{
  histogram.resize((unsigned long) size);
  for (int i = 0; i < size; ++i)
  {
    histogram[i].resize((unsigned long) size);
  }
  step = (maxValue-minValue)/size;
}

const vector<vector<int>> &JointHistogram::getHistogram() const
{
  return histogram;
}

int JointHistogram::getSize() const
{
  return size;
}

void JointHistogram::setHistogramFromPointer(int *histogram)
{
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      this->histogram[i][j] = histogram[size*i+j];
      if(this->histogram[i][j] < min)
      {
        min = this->histogram[i][j];
      }
      if(this->histogram[i][j] > max)
      {
        max = this->histogram[i][j];
      }
    }
  }
}

void JointHistogram::setColSumFromPointer(int *colSums)
{
  numValues = 0;
  for (int i = 0 ; i < size; ++i)
  {
    this->colSums[i] = colSums[i];
    numValues += colSums[i];
  }
}

void JointHistogram::setRowSumFromPointer(int *rowSums)
{
  for (int i = 0 ; i < size; ++i)
  {
    this->rowSums[i] = rowSums[i];
  }
}

void JointHistogram::setAllFromPointer(int *histogram, int *colSums, int *rowSums, int size)
{
  this->size = size;
  setHistogramFromPointer(histogram);
  setColSumFromPointer(colSums);
  setRowSumFromPointer(rowSums);
}

JointHistogram::JointHistogram()
{}

void JointHistogram::init(int size)
{
  this->size = size;
  histogram.resize((unsigned long) size);
  for (int i = 0; i < size; ++i)
  {
    histogram[i].resize((unsigned long) size);
  }
  colSums.resize((unsigned long) size);
  rowSums.resize((unsigned long) size);
}
