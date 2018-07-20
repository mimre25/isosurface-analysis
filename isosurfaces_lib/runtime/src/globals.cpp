//
// Created by mimre on 7/26/16.
//
#include "runtime/hdr/globals.h"
#include <string>
#include <stdio.h>
#include <stdarg.h>


void printfn (std::string msg, ...)
{
  if (PRINTING)
  {
    va_list myargs;
    va_start(myargs, msg);

    /* Forward the '...' to vprintf */
    vprintf((msg + "\n").c_str(), myargs);

    /* Clean up the va_list */
    va_end(myargs);
  }
};


std::string stringPrintf(std::string msg, ...)
{
  va_list myargs;
  va_start(myargs, msg);

  /* Forward the '...' to vprintf */
  char str[2<<15];//todo check this
  vsprintf(str, msg.c_str(), myargs);


  /* Clean up the va_list */
  va_end(myargs);
  std::string s = std::string(str);
  return s;
}
