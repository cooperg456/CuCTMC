#pragma once

#include "ReactionNetwork.hpp"

#define MAX_REACTANTS 16    //  fix this later
#define MAX_REACTIONS 16

struct dataOut {
    std::vector<std::vector<std::vector<int>>> paths{};
    std::vector<std::vector<double>> times{};
};

dataOut stochasticSimulation(ReactionNetwork);
