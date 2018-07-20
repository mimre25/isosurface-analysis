//
// Created by mimre on 1/18/17.
//

#include "DAO/hdr/RepresentativesInfo.h"

RepresentativesInfo::RepresentativesInfo()
{}

RepresentativesInfo::RepresentativesInfo(int valueId, float isovalue, int repId, float importance,
                                         const std::string &filename, const int mapId) : valueId(valueId), isovalue(isovalue),
                                                                        repId(repId), importance(importance),
                                                                        filename(filename), mapId(mapId)
{}
