//
// Created by mimre on 7/27/16.
//

#ifndef FILEHANDLER_FILEWRITER_H
#define FILEHANDLER_FILEWRITER_H

#include <string>
#include <vector>
#include "FileHandler.h"

namespace filehandler
{

  using namespace std;

  class FileWriter : public FileHandler
  {

  public:
    enum WriteMode
    {
      WRITE, //w
      APPEND, //a
      WRITE_READ,//w+
      BINARY_WRITE, //wb
      BINARY_APPEND, //ab
      BINARY_WRITE_READ //w+b
    };

    FileWriter(const string &fileName);

    FileWriter();

    void writeLines(vector<string> lines);


  protected:
    void evaluateFileOpenMode(WriteMode mode);

  public:
    FileWriter(const string &fileName, WriteMode writeMode);

  protected:
    WriteMode writeMode;


  public:
    WriteMode getWMode() const;

    void setWMode(WriteMode writeMode);

    void write(string s);
  };

}
#endif //FILEHANDLER_FILEWRITER_H
