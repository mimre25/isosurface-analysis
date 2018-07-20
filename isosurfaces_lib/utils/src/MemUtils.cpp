//
// Created by mimre on 1/13/17.
//

#include <sys/sysinfo.h>
#include <iostream>
#include "utils/hdr/MemUtils.h"
#include "runtime/hdr/globals.h"


//simple debug function
void MemUtils::checkmem(std::string s, bool print)
{
  struct sysinfo memInfo;
  if (DEBUG && print)
  {
    sysinfo(&memInfo);
    long long physMemUsed = memInfo.totalram - memInfo.freeram;
//Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;
    std::cout << s << ", " << physMemUsed / 1024 / 1024 << "MB in use:" << std::endl;
  }
}