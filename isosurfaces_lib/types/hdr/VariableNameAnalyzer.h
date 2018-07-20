//
// Created by mimre on 1/20/17.
//

#ifndef ISOSURFACES_VARIABLENAMEANALYZER_H
#define ISOSURFACES_VARIABLENAMEANALYZER_H


#include <string>

class VariableNameAnalyzer
{
public:
  static std::string getFileNameWithoutExtension(std::string fullPath);
  static std::string getCommonPrefix(std::string filename);
  static std::string getVariableName(std::string filename);
  static int analyzeTimeStep(std::string filename);
};


#endif //ISOSURFACES_VARIABLENAMEANALYZER_H
