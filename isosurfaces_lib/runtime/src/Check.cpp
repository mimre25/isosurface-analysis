//
// Created by mimre on 6/15/18.
//

#include <DAO/hdr/RunInformation.h>
#include <json/hdr/JSONHandler.h>
#include <utils/hdr/RuntimeInfoUtils.h>
#include "runtime/hdr/Check.h"

Check::Check(const std::string &file, int ts1, int ts2) : file(file), ts1(ts1), ts2(ts2)
{}

int Check::run()
{

  printf("Loading file\n");
  RunInformation hostFile = json::JSONHandler::loadJSON<RunInformation>(file);
  printf("Done Loading file\n");


  auto variables = hostFile.variables;
  RuntimeInfoUtils riu;
  auto hostSet = riu.createHostSet(hostFile);

  int missingCounter = 0;
  printf("Begin Check\n");
  for (int v1 = 0; v1 < variables.size(); ++v1)
  {
    for (int v2 = v1; v2 < variables.size(); ++v2)
    {
      for (int t1 = ts1; t1 < ts2; ++t1)
      {
        int oldC = missingCounter;
        //Compute var x map (eg vort x chi with same timestep)
        if (v1 != v2)
        {
          int t2 = t1;

          if (hostSet.find(entry{v1, v2, t1, t2}) == hostSet.end())
          {
            printf("Missing %s %s %d %d\n", variables[v1].c_str(), variables[v2].c_str(), t1, t2);
            ++missingCounter;
          }

        } else // different timestep, same variable (eg vort-1 x vort-2)
        {
          for (int t2 = t1 + 1; t2 < ts2; ++t2)
          {
            if (hostSet.find(entry{v1, v2, t1, t2}) == hostSet.end())
            {
              printf("Missing %s %s %d %d\n", variables[v1].c_str(), variables[v2].c_str(), t1, t2);
              ++missingCounter;
            }
          }
        }
        if (oldC == missingCounter)
        {
          printf("%s TS %d: none\n", variables[v1].c_str(), t1);
        } else
        {
          printf("%s TS %d: %d missing\n", variables[v1].c_str(), t1, missingCounter - oldC);
        }
      }
    }
  }
  printf("Total missing: %d\n", missingCounter);
  printf("Done Check\n");

}