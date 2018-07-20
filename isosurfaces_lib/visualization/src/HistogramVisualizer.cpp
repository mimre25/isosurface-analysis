//
// Created by mimre on 8/10/16.
//

#include <SDL.h>
#include <iostream>
#include <types/hdr/JointHistogram.h>
#include "visualization/hdr/HistogramVisualizer.h"
#include "runtime/hdr/globals.h"


void HistogramVisualizer::set_pixel(SDL_Surface *surface, int x, int y, Uint32 pixel)
{
  Uint32 *target_pixel = (Uint32 *) ((Uint8 *) surface->pixels + (histSize-y-1) * surface->pitch + x * sizeof *target_pixel);
  *target_pixel = pixel;
}

int HistogramVisualizer::setup()
{
  if (SDL_Init(SDL_INIT_VIDEO) != 0)
  {
    std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;

  }
  return 1;
}
void HistogramVisualizer::saveHistogramToFile(string fileName)
{
  SDL_Surface *surface = SDL_CreateRGBSurface(0, histSize+1, histSize+1, 32, rmask, gmask, bmask, amask);
  for (int j = 0; j < histSize; ++j)
  {
    for(int k = 0; k < histSize; ++k) {
      set_pixel(surface, j, k, (Uint32) (0xff000000 | vals[j*(histSize) + k]));
    }
  }



  SDL_Rect srcrect;
  SDL_Rect dstrect;

  srcrect.x = 0;
  srcrect.y = 0;
  srcrect.w = histSize;
  srcrect.h = histSize;
  dstrect.x = 0;
  dstrect.y = 0;
  dstrect.w = 640;
  dstrect.h = 480;

  SDL_Surface *dst = SDL_CreateRGBSurface(0, 480, 480, 32, rmask, gmask, bmask, amask);;

  SDL_BlitScaled(surface, &srcrect, dst, &dstrect);


  SDL_SaveBMP(dst, fileName.c_str());


  SDL_FreeSurface(dst);
  SDL_FreeSurface(surface);
//  free(vals);
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

void HistogramVisualizer::setVals(Uint32 *vals)
{
  HistogramVisualizer::vals = vals;
}


void HistogramVisualizer::setVals(vector< vector<int> > values, int min, int max)
{
  int dist = max - min;
//  dist = dist != 0 ? dist : 1;
  for (int i = 0; i < histSize; ++i)
  {
    for (int j = 0; j < histSize; ++j)
    {
      int tmp = 0xff-(0xff*values[i][j]/dist);
      vals[i*histSize+j] = (Uint32) ( tmp << 16 | tmp << 8 | tmp | 0xff000000);
    }
  }

}

HistogramVisualizer::HistogramVisualizer(int histSize, const JointHistogram &jointHistogram) : histSize(histSize)
{
  vals = (Uint32 *) malloc(histSize * histSize * sizeof(Uint32));
  setVals(jointHistogram.getHistogram(), jointHistogram.getMin(), jointHistogram.getMax());
}
