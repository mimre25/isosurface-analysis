//
// Created by mimre on 8/10/16.
//

#ifndef SDL_SCRATCH_HISTOGRAMVISUALIZER_H
#define SDL_SCRATCH_HISTOGRAMVISUALIZER_H

#include <SDL.h>
#include <SDL_quit.h>
#include <SDL_surface.h>
#include <vector>
#include <types/hdr/JointHistogram.h>

using namespace std;

class HistogramVisualizer
{
public:
  HistogramVisualizer(int histSize, const JointHistogram &jointHistogram);

  int setup();
  void saveHistogramToFile(string fileName);

private:
  void set_pixel(SDL_Surface *surface, int x, int y, Uint32 pixel);
  Uint32* vals;
  int histSize;
public:
  void setVals(vector< vector<int> > values, int min, int max);
  void setVals(Uint32 *vals);

public:
  HistogramVisualizer();


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

};


#endif //SDL_SCRATCH_HISTOGRAMVISUALIZER_H
