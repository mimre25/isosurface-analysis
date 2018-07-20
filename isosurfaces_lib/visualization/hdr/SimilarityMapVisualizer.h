//
// Created by mimre on 8/10/16.
//

#ifndef SDL_SCRATCH_SIMILARITYMAPVISUALIZER_H
#define SDL_SCRATCH_SIMILARITYMAPVISUALIZER_H

#include <SDL.h>
#include <SDL_quit.h>
#include <SDL_surface.h>
#include <vector>
#include <unordered_map>
#include "types/hdr/SimilarityMap.h"

using namespace std;

class SimilarityMapVisualizer
{
public:
  int setup();
  void show(string fileName);


private:
  void set_pixel(SDL_Surface *surface, int x, int y, Uint32 pixel);
  Uint32* vals;
  int mapSize;
  SimilarityMap* similarityMap;
  bool color;
public:

  SimilarityMapVisualizer(int mapSize, SimilarityMap *similarityMap, bool color, vector<int>& possibleValues);

public:
  void setVals(vector< vector<float> > values);
  void setVals(Uint32 *vals);

  virtual ~SimilarityMapVisualizer();


private:
  SDL_Window *win;
  SDL_Renderer *ren;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
  Uint32 rmask = 0xff000000;
  Uint32 gmask = 0x00ff0000;
  Uint32 bmask = 0x0000ff00;
  Uint32 amask = 0x000000ff;
#else
  Uint32 rmask = 0x000000ff;
  Uint32 gmask = 0x0000ff00;
  Uint32 bmask = 0x00ff0000;
  Uint32 amask = 0xff000000;
#endif

  unordered_map<int, SimilarityMap::RepInfo>* recommendations;
};


#endif //SDL_SCRATCH_SIMILARITYMAPVISUALIZER_H
