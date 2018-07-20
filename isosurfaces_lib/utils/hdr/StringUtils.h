//
// Created by mimre on 7/28/16.
//

#ifndef UTILS_STRINGUTILS_H
#define UTILS_STRINGUTILS_H

#include <string>
#include <sstream>
#include <vector>
namespace utils
{


  using namespace std;

  class StringUtils
  {
  public :
    static void split(const string &s, char delim, vector<string> &elems);

    static vector<string> split(const string &s, char delim);


  };

}
#endif //UTILS_STRINGUTILS_H
