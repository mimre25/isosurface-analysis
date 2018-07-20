//
// Created by mimre on 8/10/16.
//


#include <cstring>
#include <cmath>
#include "types/hdr/SimilarityMap.h"
#include "filehandler/hdr/FloatBinaryWriter.h"
#include "runtime/hdr/globals.h"
#include <tuple>
#include <iostream>

void SimilarityMap::create()
{
  int size = (int) histograms.size();
  similarityMap.resize((unsigned long) size);
  for(int i = 0; i < size; ++i)
  {
    similarityMap[i].resize((unsigned long) size);
    for(int j = 0; j < size; ++j)
    {
      similarityMap[i][j] = calculateMutualInformation(histograms[i][j]);
    }
  }
}

float SimilarityMap::calculateMutualInformation(JointHistogram &histogram)
{
  int size = histogram.getSize();
  float numValues = (float) histogram.getNumValues();
  vector<int> colSums = histogram.getColSums();
  vector<int> rowSums = histogram.getRowSums();
  vector< vector<int> > hist = histogram.getHistogram();
  float hX = 0;
  float hY = 0;
  float hXY = 0;
  for(int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      if (hist[i][j] > 0)
      {
        float pxy = hist[i][j]/numValues;
        hXY -= pxy * log(pxy);
      }
    }
    if (colSums[i] > 0)
    {
      float px = colSums[i]/numValues;
      hX -= px * log(px);
    }
    if (rowSums[i] > 0)
    {
      float py = rowSums[i]/numValues;
      hY -= py * log(py);
    }
  }
  float iXY = hX + hY - hXY;
  float val = 2 * iXY/(hX + hY);
  return val != val ? 0.0f : val;
}

void SimilarityMap::setHistograms(const vector<vector<JointHistogram>> &histograms)
{
  SimilarityMap::histograms = histograms;
}

const vector<vector<float>> &SimilarityMap::getSimilarityMap() const
{
  return similarityMap;
}

void SimilarityMap::print()
{
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      printf("%0.10f;", 1-similarityMap[i][j]);
    }
    printf("\n");
  }
}



void SimilarityMap::prioritize(fibonacci_heap< SimilarityMap::RepInfo >* pq, vector< SimilarityMap::RepInfo> elements, vector<handle_t>* handles)
{
  //take best element

  if (!elements.empty())
  {
    int elemSize = (int) elements.size();
    int maxIdx = -1;
    for (int l = 0; l < elemSize; ++l)
    {
      if (elements[l].mapId > maxIdx)
      {
        maxIdx = elements[l].mapId;
      }
    }

    float similarityDistribution[maxIdx+1];
    for (int i = 0; i < elemSize; ++i)
    {
      float sum = 0.0f;
      for (int j = 0; j < elemSize; ++j)
      {
        sum += similarityMap[elements[i].mapId][elements[j].mapId];
      }
      similarityDistribution[elements[i].mapId] = sum / elements.size();
    }

    //ELEMENTS NEED TO HAVE (realIndex, index, avg similarity)
    SimilarityMap::RepInfo current;
    float max = -1.0f;
    int index = -1;
    int realIndex = -1;
    float isovalue = -1.0f;
    for (SimilarityMap::RepInfo &f: elements)
    {
      if (similarityDistribution[f.mapId] > max)
      {
        current = f;
        max = similarityDistribution[f.mapId];
        index = f.mapId;
        realIndex = f.id;
        isovalue = f.isovalue;
      }
    }
    //enq
    float p = elements.size() * similarityDistribution[index];

    handles->push_back(pq->push(SimilarityMap::RepInfo {isovalue, p, realIndex, index}));

    //reccalls
    vector<SimilarityMap::RepInfo> bigger;
    vector<SimilarityMap::RepInfo> smaller;
    for (SimilarityMap::RepInfo &f: elements)
    {
      if (f.mapId < index)
      {
        smaller.push_back(f);
      } else
      {
        if (f != current)
        {
          bigger.push_back(f);
        }
      }

    }


    prioritize(pq, smaller, handles);
    prioritize(pq, bigger, handles);

  }

}

unordered_map<int, SimilarityMap::RepInfo>* SimilarityMap::selectValues(fibonacci_heap<SimilarityMap::RepInfo> *pq, vector<handle_t> *handles,
                                                       int cnt)
{
  unordered_map<int, SimilarityMap::RepInfo>* recommendatedValues = new unordered_map<int, SimilarityMap::RepInfo>();

  cout << "ISOVALUES : " << cnt << endl;

  //pop
  for (int i = 0; i < cnt; ++i)
  {
    SimilarityMap::RepInfo  elem = pq->top();
    cout << "[" << elem.id << "] " << elem.priority << ",";
    recommendatedValues->insert(pair<int, SimilarityMap::RepInfo>(i, SimilarityMap::RepInfo{elem.isovalue, elem.priority, elem.id, elem.mapId}));
    handle_t current;
    for (unsigned int j = 0; j < handles->size(); ++j)
    {
      handle_t handle = handles->at((unsigned long) j);
      if ((*handle) != elem)
      {
        float p = (*handle).priority;
        int idx = (*handle).mapId;
        p /= (1 + similarityMap[elem.mapId][idx]);
        pq->update(handle, SimilarityMap::RepInfo{elem.isovalue, p, (*handle).id, idx});
      } else {
        current = handle;
      }
    }
    cout << endl;
    pq->pop();
    handles->erase(std::remove(handles->begin(), handles->end(), current), handles->end());
  }
  cout << endl;
  return recommendatedValues;

}


/**
 * This method implements Bruckner et al's algorithm to select representative isovalues from the similarity map
 * @param cnt number of isovalues to select
 * @param possibleSurfaces output vector for ids of the surfaces
 * @return a map with id and representative informations (~)
 */
unordered_map<int, SimilarityMap::RepInfo> * SimilarityMap::findRepresentativeIsovalues(int cnt, const vector<int> &possibleSurfaces)
{
  if (RECOMMENDATIONS)
  {
    cout << cnt << endl << endl;
    vector<SimilarityMap::RepInfo> elements;
    for (int i = 0; i < size; ++i)
    {
      elements.push_back(SimilarityMap::RepInfo{isovalues[possibleSurfaces[i]], -1.0f, possibleSurfaces[i],
                                                i});//it's  IsoValue, priority, realIndex, mapIndex,
    }
    //ELEMENTS HAVE (index,, mapIndex isovalues)
    fibonacci_heap<SimilarityMap::RepInfo> pq;
    vector<handle_t> handles;
    cout << elements.size() << endl;
    prioritize(&pq, elements, &handles);
    recommendedVals = selectValues(&pq, &handles, cnt);

    return recommendedVals;
  }
  return NULL;
}

void SimilarityMap::setIsovalues(float *isovalues)
{
  this->isovalues = isovalues;
}

SimilarityMap::SimilarityMap(float *similarityMap, const int size) : size(size)
{
  this->similarityMap.resize((unsigned long) size);
  for (int i = 0; i < size; ++i)
  {
    this->similarityMap[i].resize((unsigned long) size);
    for (int j = 0; j < size; ++j)
    {
      this->similarityMap[i][j] = similarityMap[i * size + j];
    }
  }
  pointerSet = false;


}

SimilarityMap::SimilarityMap() : size(-1)
{
}

void SimilarityMap::save(string fileName)
{
  filehandler::FloatBinaryWriter floatBinaryWriter(fileName);
  floatBinaryWriter.writeFile(fileName, getMapAsPointer(), similarityMap.size() * similarityMap[0].size());
  free(mapPointer);
  pointerSet = false;
}

float *SimilarityMap::getMapAsPointer()
{
  if(!pointerSet)
  {
    long len = similarityMap.size();
    long inLen = similarityMap[0].size();
    mapPointer = (float *) malloc((size_t) (len * inLen)* sizeof(float));
    for (int i = 0; i < len; ++i)
    {
      for (int j = 0; j < inLen; ++j)
      {
        mapPointer[i*inLen+j] = similarityMap[i][j];
      }
    }
    pointerSet = true;
  }
  return mapPointer;
}

SimilarityMap::SimilarityMap(const vector<vector<float>> &similarityMap, int size) : similarityMap(similarityMap),
                                                                                     size(size)
{}

