//
// Created by mimre on 7/28/16.
//

#include "utils/hdr/StringUtils.h"
namespace utils
{
  void StringUtils::split(const string &s, char delim, vector<string> &elems)
  {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim))
    {
      elems.push_back(item);
    }
  }


  vector<string> StringUtils::split(const string &s, char delim)
  {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
  }
}