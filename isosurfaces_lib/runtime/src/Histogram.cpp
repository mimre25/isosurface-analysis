//
// Created by mimre on 6/21/18.
//

#include <types/hdr/DistanceField.h>
#include <types/hdr/SimilarityMap.h>
#include <isosurfaces_cuda/funcs.h>
#include "runtime/hdr/Histogram.h"
#include "utils/hdr/Report.h"

Histogram::Histogram(const vector<string> &files, int dimX, int dimY, int dimZ) : files(files), dimX(dimX), dimY(dimY),
                                                                                  dimZ(dimZ)
{}
SimilarityMap computeHistogram(std::vector<DistanceField> fields, int size)
{
  vector<vector<JointHistogram> > histograms;
  int histogramSize = HIST_SIZE;

  string similarityMapCalc = "similarity map";
  Report::begin(similarityMapCalc);
  SimilarityMap similarityMap = calculate_histogram_W(fields, size, histogramSize, fields.size(), false);
  Report::end(similarityMapCalc);

  return similarityMap;
}

int Histogram::run()
{
  printf("loading files\n");
  std::vector<DistanceField> fields;
  size_t size = static_cast<size_t>(dimX * dimY * dimZ);
  for (int i = 0; i < files.size(); ++i)
  {
    auto f = files[i];
    printf("file %d... ", i);
    DistanceField df;
    df.loadFromFile(f, size);
    fields.push_back(df);
  }
  printf("finish loading files\n");

  printf("starting joint histogram computation\n");
  Report::begin("Simmap");
  SimilarityMap similarityMap = computeHistogram(fields, size);
  Report::end("Simmap");
  printf("done joint histogram\n");

  printf("saving Similarity map\n");
  similarityMap.save("test.map");
  printf("done saving map\n");

}

