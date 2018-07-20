//
// Created by mimre on 7/27/16.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "filehandler/hdr/FileWriter.h"
#include <boost/algorithm/string/join.hpp>
namespace filehandler
{
  void FileWriter::writeLines(vector<string> lines)
  {
    evaluateFileOpenMode(writeMode);
    cout << "writing file " << fileName << endl;
    ofstream file(fileName.c_str(), fileOpenMode);
    stringstream stream;

    string joinedString = boost::algorithm::join(lines, "\n");


    file << joinedString;
    file.close();
    cout << "done" << endl;
  }


  void FileWriter::evaluateFileOpenMode(FileWriter::WriteMode mode)
  {
    switch (mode)
    {
      case WriteMode::WRITE:
        fileOpenMode = ios::out | ios::trunc;
        break;
      case WriteMode::APPEND:
        fileOpenMode = ios::out | ios::app;
        break;
      case WriteMode::WRITE_READ:
        fileOpenMode = ios::in | ios::out | ios::trunc;
        break;
      case WriteMode::BINARY_WRITE:
        fileOpenMode = ios::out | ios::trunc | ios::binary;
        break;
      case WriteMode::BINARY_APPEND:
        fileOpenMode = ios::out | ios::app | ios::binary;
        break;
      case WriteMode::BINARY_WRITE_READ:
        fileOpenMode = ios::in | ios::out | ios::trunc | ios::binary;
        break;
      default:
        fileOpenMode = ios::out;
        break;
    }
  }

  FileWriter::WriteMode FileWriter::getWMode() const
  {
    return writeMode;
  }

  void FileWriter::setWMode(FileWriter::WriteMode writeMode)
  {
    FileWriter::writeMode = writeMode;
  }

  FileWriter::FileWriter(const string &fileName) : FileHandler(fileName)
  {
    writeMode = WRITE;
  }

  FileWriter::FileWriter(const string &fileName, FileWriter::WriteMode writeMode) : FileHandler(fileName),
                                                                                    writeMode(writeMode)
  {}

  void FileWriter::write(string s)
  {
    evaluateFileOpenMode(writeMode);
    ofstream file(fileName.c_str(), fileOpenMode);
    file << s;
    file.close();
  }

  FileWriter::FileWriter()
  {
    writeMode = WRITE;
  }



}