//
// Created by mimre on 7/27/16.
//

#ifndef FILEHANDLER_FILEREADER_H
#define FILEHANDLER_FILEREADER_H

#include <string>
#include <vector>
#include <bits/ios_base.h>
#include "FileHandler.h"

namespace filehandler
{
  using namespace std;


  class FileReader : public FileHandler
  {
  public:
    enum ReadMode
    {
      READ, //r
      READ_WRITE, //r+
      BINARY_READ, //rb
      BINARY_READ_WRITE //r+b
    };

    FileReader(const string &fileName);

    FileReader(const string &fileName, ReadMode readMode);

    FileReader();

    unsigned char *get_file_contents();

    string getFileContentsAsString();

    ReadMode getReadMode() const;

    vector<string> readLines();
    void setReadMode(ReadMode readMode);
  private:

    ReadMode readMode;

  protected:
    void evaluateFileOpenMode(ReadMode mode);
  };
}

#endif //FILEHANDLER_FILEREADER_H
