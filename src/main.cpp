#include <iostream>
#include <fstream>

#include "cuda.cuh"
#include "ReactionNetwork.hpp"
 
int main(int argc, char *argv[]) {
	if (argc != 2) {
		throw std::runtime_error("json file not provided");
	}

    ReactionNetwork cme(argv[1]);

    dataOut output = stochasticSimulation(cme);

    std::ofstream file("paths.csv");
    if (file.is_open()) {
        size_t reactants = cme.initialConditions.size();

        file << "path,step,time";
        for (size_t r = 0; r < reactants; r++) {
            file << ",reactant_" << r;
        }
        file << "\n";

        for (size_t path = 0; path < output.paths.size(); path++) {
            for (size_t step = 0; step < output.paths[path].size(); step++) {
                file << path << "," << step << "," << output.times[path][step];
                for (size_t r = 0; r < reactants; r++) {
                    file << "," << output.paths[path][step][r];
                }
                file << "\n";
            }
        }
        file.close();
    } else {
        std::cerr << "Failed to open paths.csv for writing\n";
    }
        
    return 0;
}
