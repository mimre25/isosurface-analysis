//
// Created by mimre on 6/7/17.
//

#ifndef ISOSURFACES_DISABLEPRINT_H
#define ISOSURFACES_DISABLEPRINT_H



//#define DISABLE_PRINTF

#ifdef DISABLE_PRINTF
  #define printf(fmt, ...) (0)
#endif



#endif //ISOSURFACES_DISABLEPRINT_H
