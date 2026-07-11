#include <iostream>
#include <fstream>

#include "StochasticSimulation.cuh"
#include "ReactionNetwork.hpp"
 
int main(int argc, char *argv[]) {
	if (argc != 2) {
		throw std::runtime_error("json file not provided");
	}

    ReactionNetwork network(argv[1]);

    SSASimInfo info{};
    info.initialConditions = {1000, 3, 0, 0};
    info.samplePaths = 256;

    SSASimOutput out = stochasticSimulation(network, info);

    return 0;
}
