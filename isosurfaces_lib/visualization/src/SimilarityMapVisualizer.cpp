//
// Created by mimre on 8/10/16.
//

#include <SDL.h>
#include <iostream>
#include <runtime/hdr/DistanceEqualization.h>
#include "visualization/hdr/SimilarityMapVisualizer.h"
#include "runtime/hdr/globals.h"


void SimilarityMapVisualizer::set_pixel(SDL_Surface *surface, int x, int y, Uint32 pixel)
{
  Uint32 *target_pixel = (Uint32 *) ((Uint8 *) surface->pixels + (mapSize-y-1) * surface->pitch + x * sizeof *target_pixel);
  *target_pixel = pixel;
}

int SimilarityMapVisualizer::setup()
{
  if (SDL_Init(SDL_INIT_VIDEO) != 0){
    std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;

  }
  return 1;
}

void SimilarityMapVisualizer::show(string fileName)
{
  SDL_Surface *surface = SDL_CreateRGBSurface(0, mapSize+1, mapSize+1, 32, rmask, gmask, bmask, amask);
  for (int j = 0; j < mapSize; ++j)
  {
    for(int k = 0; k < mapSize; ++k) {
      set_pixel(surface, j, k, (Uint32) (0xff000000 | vals[j*(mapSize) + k]));
    }
  }


  cout << fileName << endl;


  SDL_Rect srcrect;
  SDL_Rect dstrect;

  srcrect.x = 0;
  srcrect.y = 0;
  srcrect.w = mapSize;
  srcrect.h = mapSize;
  dstrect.x = 0;
  dstrect.y = 0;
  dstrect.w = 640;
  dstrect.h = 480;

  SDL_Surface *dst = SDL_CreateRGBSurface(0, 480, 480, 32, rmask, gmask, bmask, amask);;

  SDL_BlitScaled(surface, &srcrect, dst, &dstrect);


  SDL_SaveBMP(dst, fileName.c_str());
  free(vals);
  SDL_FreeSurface(dst);
  SDL_FreeSurface(surface);

  SDL_Event event;
  bool quit=true;
  while(!quit)
  {
    while(SDL_PollEvent(&event))
    {
      if(event.type == SDL_QUIT)
      {
        quit = true;
      }
    }
  }

}

void SimilarityMapVisualizer::setVals(Uint32 *vals)
{
  SimilarityMapVisualizer::vals = vals;
}



void SimilarityMapVisualizer::setVals(vector< vector<float> > values)
{
  priority_queue<SimilarityMap::RepInfo> pq = priority_queue<SimilarityMap::RepInfo>();
  unordered_map<int, int> indexColorMap = unordered_map<int,int>();
  if (RECOMMENDATIONS)
  {
  int colors[8] = { 0xe0ffff,0xa3daff,0x7cb2ff,0x6887fb,0x5b5feb,0x4a39d3,0x2f15b3,0x00008b};
  for(auto kv : (*recommendations))
  {
    pq.push(SimilarityMap::RepInfo{kv.second.isovalue, kv.second.priority, kv.second.id, kv.second.mapId});
  }
  for (int i = 0; i < 8; ++i)
  {
    SimilarityMap::RepInfo elem = pq.top();
    pq.pop();
    indexColorMap.insert(pair<int,int>(elem.mapId, colors[i]));
  }
  }

  unordered_map<int, int>::iterator endPtr = indexColorMap.end();
  for (int i = 0; i < mapSize; ++i)
  {
    for (int j = 0; j < mapSize; ++j)
    {
      int tmp = (int) (0xff - (0xff * values[i][j]));
      if (color && i == j && indexColorMap.find(j) != endPtr)
      {
        int col = indexColorMap.at(j);

        vals[i*mapSize+j] = (Uint32) (0xff000000 | col );
      } else {
        vals[i*mapSize+j] = (Uint32) ( tmp << 16 | tmp << 8 | tmp | 0xff000000);
      }
    }
  }

}


SimilarityMapVisualizer::~SimilarityMapVisualizer()
{
}

SimilarityMapVisualizer::SimilarityMapVisualizer(int mapSize, SimilarityMap *similarityMap, bool color, vector<int>& possibleValues) : mapSize(
    mapSize), similarityMap(similarityMap), color(color)
{
  vals = (Uint32 *) malloc(mapSize * mapSize * sizeof(Uint32));
  recommendations = similarityMap->findRepresentativeIsovalues(NUM_DISTANCES, possibleValues);
  setVals(similarityMap->getSimilarityMap());
}


