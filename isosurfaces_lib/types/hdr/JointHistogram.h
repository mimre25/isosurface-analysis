//
// Created by mimre on 8/8/16.
//

#ifndef ISOSURFACES_JOINTHISTOGRAM_H
#define ISOSURFACES_JOINTHISTOGRAM_H


#include "DistanceField.h"
#include <vector>

using namespace std;
class JointHistogram
{
public:
  void calculateHistogram(DistanceField* field1, DistanceField* field2);

  int findBucket(float val);

  JointHistogram();

  JointHistogram(int size, int min, int max);

  void print();

  const vector< vector<int> > &getHistogram() const;

  int getMin() const;

  int getMax() const;

  const vector<int> &getRowSums() const;

  const vector<int> &getColSums() const;

  int getSize() const;

  int getNumValues() const;

  void setHistogramFromPointer(int* histogram);
  void setColSumFromPointer(int* colSums);
  void setRowSumFromPointer(int* rowCums);

  void setAllFromPointer(int *histogram, int *colSums, int *rowSums, int size);

  void init(int size);

protected:
  int size;
  vector< vector<int> > histogram;
  int min;
  int max;
  float minValue;
  float maxValue;
  float step;
  vector<int> rowSums;
  vector<int> colSums;

  int numValues;
};


#endif //ISOSURFACES_JOINTHISTOGRAM_H
