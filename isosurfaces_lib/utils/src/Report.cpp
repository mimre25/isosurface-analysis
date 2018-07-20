//
// Created by mimre on 8/15/16.
//
#include <boost/algorithm/string/predicate.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "utils/hdr/Report.h"
map<string,clock_t> Report::entries;
map<string,double> Report::totals;
Report* Report::instance;





void Report::output (string s)
{

  if (Report::getInstance("")->writeToFile)
  {
    Report::getInstance("")->fileWriter->write(s);
  }
  else
  {
    cout << s;
  }
}

void Report::begin(string name)
{
  entries[name] = clock();

  output(name + "... ");
}

void Report::end(string name)
{
  clock_t end = clock();
  clock_t begin = entries[name];
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  output(formatTime(elapsed_secs) + "\n");
  totals[name] += elapsed_secs;
  entries.erase(name);
}

string Report::formatTime(double seconds)
{
  int hours = 0;
  int minutes = 0;
  int secs = (int) seconds;
  stringstream result;
  if (secs > 3600)
  {
    hours = secs/3600;
    secs %= 3600;
  }
  if (secs > 60)
  {
    minutes = secs/60;
    secs %= 60;
  }
  result << setw(2) << setfill('0') << hours;
  result << ":";
  result << setw(2) << setfill('0') << minutes;
  result << ":";
  result << setw(2) << setfill('0') << secs;
  result << "\t" << seconds << "seconds";
  return result.str();
}

void Report::beginQuiet(string name)
{
  entries[name] = clock();
  if (totals.find(name) == totals.end())
  {
    totals[name] = clock();
  }
}

void Report::endQuiet(string name)
{
  output(name + "... ");
  end(name);
}

void Report::printTotals(bool csv)
{
  output("\n\nSUMMARY\n\n");
  for (map<string,double>::iterator it = totals.begin(); it != totals.end(); ++it)
  {
    output(it->first + ": " + formatTime(it->second) + "\n");
  }
  if (csv)
  {
    stringstream s;
    for (map<string,double>::iterator it = totals.begin(); it != totals.end(); ++it)
    {
      output(it->first + ";");
      s << setprecision(5) << it->second << ";";
    }
    output("\n");
    output(s.str() + "\n");

  }
}


Report *Report::getInstance(string fileName)
{
  if (!instance)
    instance = new Report(fileName);
  return instance;
}

Report::Report(const string &outPutFile)
{
  this->fileWriter = new filehandler::FileWriter(outPutFile, filehandler::FileWriter::APPEND);
  writeToFile = outPutFile != "";
}

void Report::clear()
{
  Report::entries.clear();
  Report::totals.clear();
  Report::entries = map<string,clock_t>();
  Report::totals = map<string,double>();
}


double Report::getRuntime(string name)
{
  clock_t end = clock();
  clock_t begin = (clock_t) totals[name];
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  return elapsed_secs;

}

string Report::getTotals()
{
  stringstream s;
  double estimationStage = 0;
  double refinementStage = 0;


  for (map<string,double>::iterator it = totals.begin(); it != totals.end(); ++it)
  {
    if(boost::starts_with(it->first, "REFINEMENT-"))
    {
      refinementStage += it->second;
    }
    if(boost::starts_with(it->first, "ESTIMATION-"))
    {
      estimationStage += it->second;
    }
    s << setprecision(5) << it->second << ",";
  }
  s << setprecision(5) << estimationStage << "," << setprecision(5) << refinementStage << ",";
  return s.str();
}

string Report::getTotalsNames()
{
  stringstream s;
  for (map<string,double>::iterator it = totals.begin(); it != totals.end(); ++it)
  {
    s << it->first << ",";
  }
  s << "ESTIMATION-Stage" << "," << "REFINEMENT-Stage" << ",";
  return s.str();
}



