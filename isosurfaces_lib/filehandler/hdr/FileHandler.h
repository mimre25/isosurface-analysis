//
// Created by mimre on 7/27/16.
//

#ifndef FILEHANDLER_FILEHANDLER_H
#define FILEHANDLER_FILEHANDLER_H

#include <string>
#include <vector>
#include <bits/ios_base.h>

namespace filehandler
{

  using namespace std;

  class FileHandler
  {
  public:
    FileHandler(const string &fileName);

    FileHandler();
    const string &getFileName() const;

    void setFileName(const string &fileName);

  protected:
    _Ios_Openmode fileOpenMode;
    string fileName;

  };

}
#endif //FILEHANDLER_FILEHANDLER_H
