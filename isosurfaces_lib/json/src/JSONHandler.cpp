//
// Created by mimre on 1/18/17.
//

#include "json/hdr/JSONHandler.h"
#include <ThorSerialize/JsonThor.h>
#include <sstream>
#include <zconf.h>
#include "ThorSerialize/SerUtil.h"
#include "filehandler/hdr/FileWriter.h"
#include "filehandler/hdr/FileReader.h"
#include "DAO/hdr/ScriptingInput.h"
#include "DAO/hdr/SingleConfig.h"
#include "DAO/hdr/MultiVarConfig.h"
#include <ThorSerialize/Traits.h>

ThorsAnvil_MakeTrait(SimilarityMapInformation, var1, t1, var2, t2, filename);
ThorsAnvil_MakeTrait(RepresentativesInfo, valueId, isovalue, repId, importance, filename, mapId);
ThorsAnvil_MakeTrait(VolumeInformation, variable, timestep, numIsovalues, representatives, runtime);
ThorsAnvil_MakeTrait(RunInformation, dimensions, variables,  dfDownscale, volumes, similarityMaps);
ThorsAnvil_MakeTrait(VariableInfo, name, minValue, maxValue, variableFolder);
ThorsAnvil_MakeTrait(DataInfo, dimensions, variables, timesteps, dfDownscale, fileFormatString, inputFolder, outputFolder, jsonFile);
ThorsAnvil_MakeTrait(ScriptingInput, entries);
ThorsAnvil_MakeTrait(SingleConfig, fileName, dimensions, minValue, maxValue, outputFolder, variableName, timestep, dfDownscale, jsonFile);
ThorsAnvil_MakeTrait(MultiVarConfig, jsonFile, timeSteps, dataRoot, dfDownscale);

namespace json
{
  using ThorsAnvil::Serialize::jsonExport;
  using ThorsAnvil::Serialize::jsonImport;
  using ThorsAnvil::Serialize::PrinterInterface;


  template <class T> void JSONHandler::saveJSON(T object, std::string filename)
  {
    std::stringstream str;
    str << jsonExport(object);

    filehandler::FileWriter fileWriter(filename);
    fileWriter.write(str.str());
  }

  template <class T> T JSONHandler::loadJSON(std::string filename)
  {
    T object = T();
    if (( access( filename.c_str(), F_OK ) != -1 ))
    {
      filehandler::FileReader fileReader(filename);
      std::stringstream str(fileReader.getFileContentsAsString(), std::ios_base::in);
      str >> jsonImport(object, ThorsAnvil::Serialize::ParserInterface::ParseType::Strict);
    } else
    {
      printf("WARNING: file %s not found", filename.c_str());
    }
    return object;
  }


  template RunInformation JSONHandler::loadJSON(std::string);
  template void JSONHandler::saveJSON(RunInformation, std::string);
  template ScriptingInput JSONHandler::loadJSON(std::string);
  template void JSONHandler::saveJSON(ScriptingInput, std::string);
  template SingleConfig JSONHandler::loadJSON(std::string);
  template void JSONHandler::saveJSON(SingleConfig, std::string);
  template MultiVarConfig JSONHandler::loadJSON(std::string);
  template void JSONHandler::saveJSON(MultiVarConfig, std::string);
}



