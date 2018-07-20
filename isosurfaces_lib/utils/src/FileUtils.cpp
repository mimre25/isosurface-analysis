//
// Created by mimre on 5/17/17.
//

#include <sys/stat.h>
#include "utils/hdr/FileUtils.h"

bool FileUtils::fileExists(const std::string &filename)
{
  {
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
      return true;
    }
    return false;
  }
}
