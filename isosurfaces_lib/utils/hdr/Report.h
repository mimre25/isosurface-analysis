//
// Created by mimre on 8/15/16.
//

#ifndef ISOSURFACES_REPORT_H
#define ISOSURFACES_REPORT_H

#include <string>
#include <map>
#include "filehandler/hdr/FileWriter.h"

#include "utils/hdr/disablePrint.h"
using namespace std;



class Report
{

public:

  template <class T>
  friend void operator<< (const Report& r, const T& t)
  {
    Report::output(t.str());
  }

  static void begin(string name);
  static void end(string name);

  static void beginQuiet(string name);

  static void endQuiet(string name);

  static void printTotals(bool csv = false);

  Report(const string &outPutFile);
  static string getTotalsNames();


  Report static *instance;

  static const string& outPutFile;

  filehandler::FileWriter *fileWriter;
  bool writeToFile;

  static Report* getInstance(string fileName);

  static void clear();

  static double getRuntime(string name);

  static string getTotals();


private:
  static map<string, clock_t> entries;
  static map<string, double> totals;

  static string formatTime(double seconds);


  static void output(string);
};


#endif //ISOSURFACES_REPORT_H
