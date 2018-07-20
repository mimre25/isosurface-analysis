//
// Created by mimre on 1/18/17.
//

#include "DAO/hdr/VolumeInformation.h"


VolumeInformation::VolumeInformation()
{}

VolumeInformation::VolumeInformation(int variable, int timestep, int numIsovalues, double runtime,
                                     const std::vector<RepresentativesInfo> &representatives) : variable(variable),
                                                                                                timestep(timestep),
                                                                                                numIsovalues(
                                                                                                    numIsovalues),
                                                                                                runtime(runtime),
                                                                                                representatives(
                                                                                                    representatives)
{}
