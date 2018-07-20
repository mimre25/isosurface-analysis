//
// Created by mimre on 7/27/16.
//

#include <bits/ios_base.h>
#include <ios>
#include <fstream>
#include <iterator>
#include <memory.h>
#include <iostream>
#include "filehandler/hdr/FileReader.h"
namespace filehandler
{
  void FileReader::evaluateFileOpenMode(FileReader::ReadMode mode)
  {
    switch (mode)
    {
      case ReadMode::READ:
        fileOpenMode = ios::in;
        break;
      case ReadMode::READ_WRITE:
        fileOpenMode = ios::in | ios::out;
        break;
      case BINARY_READ:
        fileOpenMode = ios::in | ios::binary;
        break;
      case BINARY_READ_WRITE:
        fileOpenMode = ios::in | ios::out | ios::binary;
      default:
        fileOpenMode = ios::in;
        break;
    }
  }


  unsigned char *FileReader::get_file_contents()
  {
    evaluateFileOpenMode(readMode);
    const char *fN = fileName.c_str();
    std::ifstream in(fN, fileOpenMode);
    if (in)
    {
      in.seekg(0, std::ios::end);
      long int fileSize = in.tellg();
      unsigned char *contents = (unsigned char *) malloc((size_t) (fileSize+1));
      in.seekg(0, std::ios::beg);
      in.read((char *) contents, fileSize);
      in.close();
      contents[fileSize] = '\0';
      return (contents);
    }
    perror(strerror(errno));
    throw (strerror(errno));
  }

  vector<string> FileReader::readLines()
  {
    vector<string> myLines;
    ifstream myfile(fileName);
    string line;
    while (getline(myfile, line))
    {
      myLines.push_back(line);
    }
    return myLines;
  }



  FileReader::ReadMode FileReader::getReadMode() const
  {
    return readMode;
  }

  void FileReader::setReadMode(FileReader::ReadMode readMode)
  {
    FileReader::readMode = readMode;
  }

  FileReader::FileReader(const string &fileName) : FileHandler(fileName)
  {
    readMode = READ;
  }

  FileReader::FileReader(const string &fileName, FileReader::ReadMode readMode) : FileHandler(fileName),
                                                                                  readMode(readMode)
  {}

  FileReader::FileReader()
  {
    readMode = READ;
  }

  string FileReader::getFileContentsAsString()
  {
    char* data = (char*)get_file_contents();
    std::string s(data);
    free(data);
    return s;
  }

}