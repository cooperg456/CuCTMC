#pragma once

#include "ReactionNetwork.hpp"

#define MAX_SSA_REACTANTS 60    //  sized for CUDA shared memory limits
#define MAX_SSA_REACTIONS 96
#define SSA_BLOCK_SIZE 32

struct SSASysInfo {
    std::vector<double> reactionRates{};
    std::vector<int> initialConditions{};
    std::vector<int> reactantCoefficients{};
    std::vector<int> transitionCoefficients{};    
};

struct SSASimInfo {
    double tMax = 0;
    int warps = 0;
    
    double tGrid = 0;
    int savedPaths = 0;
    std::filesystem::path outputDir{};

    ReactionNetwork base{};
};

void SSA(SSASimInfo simInfo, std::vector<SSASysInfo> &sysInfos);
