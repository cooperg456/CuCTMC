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
        0.7 / 1000, 
        1.5 * 25, 
        1.5,
        1.7, 
        4 / 1000
    };

    ChemicalMasterEquation buddingCTMC(alpha, beta, rates);

    std::vector<int> start = {
        1000, 3, 0, 0
    };

    int paths = 256;
    int steps = 1 << 16;

    dataOut output = stochasticSimulation(buddingCTMC.get_d_alpha(),
                                          buddingCTMC.get_d_trans(),
                                          buddingCTMC.get_d_rates(),
                                          start.data(),
                                          start.size(),
                                          rates.size(),
                                          paths, steps);

    std::ofstream file("path_0.csv");
    if (file.is_open()) {
        size_t reactants = start.size();

        file << "step,time";
        for (size_t r = 0; r < reactants; r++) file << ",reactant_" << r;
        file << "\n";

        for (size_t step = 0; step < output.paths[0].size(); step++) {
            file << step << "," << output.times[0][step];
            for (size_t r = 0; r < reactants; r++) {
                file << "," << output.paths[0][step][r];
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Failed to open path_0.csv for writing\n";
    }
        
    return 0;
}
