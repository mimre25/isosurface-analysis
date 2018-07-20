//
// Created by mimre on 8/10/16.
//

#ifndef ISOSURFACES_SIMILARITYMAP_H
#define ISOSURFACES_SIMILARITYMAP_H


#include <boost/heap/fibonacci_heap.hpp>
#include "JointHistogram.h"
#include "runtime/hdr/globals.h"
#include <boost/heap/fibonacci_heap.hpp>
#include <unordered_map>

using namespace boost::heap;
class SimilarityMap
{
public:
  struct RepInfo
  {
    float isovalue;
    float priority;
    int id;
    int mapId;

    bool operator==(const RepInfo& rhs)
    {
      return isovalue == rhs.isovalue && priority == priority && id == rhs.id && mapId == rhs.mapId;
    }
    bool operator!=(const RepInfo& rhs)
    {
      return isovalue != rhs.isovalue || priority != priority || id != rhs.id || mapId != rhs.mapId;
    }
    bool operator<(const RepInfo& o) const
    {
      return priority < o.priority;
    }
  };


protected:
  typedef typename fibonacci_heap< RepInfo >::handle_type handle_t;
  vector< vector<JointHistogram> > histograms;
  vector< vector<float > > similarityMap;
public:
  SimilarityMap(const vector< vector<float> > &similarityMap, int size);

protected:
  float* isovalues;
  unordered_map<int, RepInfo>* recommendedVals;
  int size;
  bool pointerSet;
  float* mapPointer;

public:
  SimilarityMap(float *similarityMap, const int size);

  SimilarityMap();

  void save(string fileName);

public:
  void create();

  void setHistograms(const vector<vector<JointHistogram> > &histograms);

  float calculateMutualInformation(JointHistogram &histogram);

  const vector<vector<float> > &getSimilarityMap() const;

  void print();

  unordered_map<int, RepInfo> * findRepresentativeIsovalues(int cnt, const vector<int> &possibleSurfaces);


  unordered_map<int, RepInfo>* selectValues(fibonacci_heap<RepInfo> *pq, vector<handle_t> *handles, int cnt = MULTI_MAP_DF_NUMBER);

  void prioritize(boost::heap::fibonacci_heap<RepInfo> *pq, vector<RepInfo> elements,
                  vector<handle_t> *handles);

  void setIsovalues(float *isovalues);

  bool representativeCalculated = false;

  float* getMapAsPointer();
};


#endif //ISOSURFACES_SIMILARITYMAP_H
