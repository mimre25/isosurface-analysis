//
// Created by mimre on 1/20/17.
//

#include <boost/filesystem.hpp>
#include "types/hdr/VariableNameAnalyzer.h"
#include "boost/regex.hpp"
#include "runtime/hdr/globals.h"


using namespace std;
string VariableNameAnalyzer::getCommonPrefix(std::string filename)
{
  char splitChars[] = { '_', '0'};
  stringstream ss(filename);
  string item;
  vector<string> tokens;
  for (char c : splitChars)
  {
    while (getline(ss, item, c))
    {
      if (item != filename)
      {
        tokens.push_back(item);
      }
    }
    ss = stringstream(filename);
  }

  return tokens.size() > 0 ? tokens[0] : filename;
}

int VariableNameAnalyzer::analyzeTimeStep(std::string filename)
{
  string s = boost::regex_replace(
      string(filename),
      boost::regex("[^0-9]*([0-9]+).*"),
      string("\\1")
  );
  int i = 0;
  if(s != filename)
  {
    i = std::stoi(s);
  }
  return i;


}

std::string VariableNameAnalyzer::getVariableName(std::string filename)
{
  char splitChars[] = { '-', '_', '0'};
  stringstream ss(filename);
  string item;
  vector<string> tokens;
  for (char c : splitChars)
  {
    while (getline(ss, item, c))
    {
      if (item != filename)
      {
        tokens.push_back(item);
      }
    }
    ss = stringstream(filename);
  }

  return tokens.size() > 0 ? tokens[0] : filename;
}

std::string VariableNameAnalyzer::getFileNameWithoutExtension(std::string fullPath)
{
  std::vector<std::string> result;
  return boost::filesystem::change_extension(stringPrintf(basename(fullPath.c_str())), "").string();

}
