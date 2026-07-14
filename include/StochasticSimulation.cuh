#pragma once

#include "ReactionNetwork.hpp"

#define MAX_SSA_REACTANTS 60    //  sized for CUDA shared memory limits
#define MAX_SSA_REACTIONS 96

struct SSASimInfo {
    //  system parameters

    std::vector<int> initialConditions{};

    //  sim parameters

    int samplePaths = 0;
    int savedPaths = 0;

    double tGrid = 0;
    double tMax = 0;
};

class SSASimOutput {    //  this class is poorly written. fix this later
public:
    SSASimOutput(double *timePoints, int *samplePaths, ReactionNetwork &network, SSASimInfo &simInfo);
    ~SSASimOutput();

    void toCSV(const std::filesystem::path& csvFile) const;

private:
    double *d_timePoints;
    int *d_samplePaths;

    ReactionNetwork &network;
    SSASimInfo &info;
};

SSASimOutput stochasticSimulation(ReactionNetwork &network, SSASimInfo &info);
