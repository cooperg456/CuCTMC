#include <iostream>
#include <fstream>

#include "cuda.cuh"
#include "cme.hpp"
 
int main(void)
{
    std::vector<std::vector<int>> alpha = {
    //   H, V, I, D
        {1, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 1}
    };

    std::vector<std::vector<int>> beta = {
    //   H, V, I, D
        {0, 0, 1, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 1},
        {0, 0, 0, 0},
        {2, 0, 0, 0}
    };

    std::vector<double> rates = {
        0.7 / 1000.0f, 
        1.5 * 25, 
        1.5,
        1.7, 
        4 / 1000.0f
    };

    ChemicalMasterEquation buddingCTMC(alpha, beta, rates);

    std::vector<int> start = {
        1000, 3, 0, 0
    };

    int paths = 256;
    int steps = 200000;

    dataOut output = stochasticSimulation(buddingCTMC.get_d_alpha(),
                                          buddingCTMC.get_d_trans(),
                                          buddingCTMC.get_d_rates(),
                                          start.data(),
                                          start.size(),
                                          rates.size(),
                                          paths, steps);

    std::ofstream file("paths.csv");
    if (file.is_open()) {
        size_t reactants = start.size();

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
