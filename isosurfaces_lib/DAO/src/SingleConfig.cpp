//
// Created by mimre on 2/3/17.
//

#include "DAO/hdr/SingleConfig.h"


SingleConfig::SingleConfig(const std::string &fileName, const std::vector<int> &dimensions, float minValue,
                           float maxValue) : fileName(fileName), dimensions(dimensions), minValue(minValue),
                                             maxValue(maxValue)
{}

SingleConfig::SingleConfig()
{}
