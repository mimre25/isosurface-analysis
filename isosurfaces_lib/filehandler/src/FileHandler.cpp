//
// Created by mimre on 7/27/16.
//

#include "filehandler/hdr/FileHandler.h"
namespace filehandler {
  const string &FileHandler::getFileName() const
  {
    return fileName;
  }

  void FileHandler::setFileName(const string &fileName)
  {
    FileHandler::fileName = fileName;
  }

  FileHandler::FileHandler(const string &fileName) : fileName(fileName)
  {}

  FileHandler::FileHandler()
  {}

}