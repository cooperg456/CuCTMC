#include <iostream>
#include <fstream>
#include <tuple>

#include "CLI/CLI11.hpp"

#include "StochasticSimulation.cuh"
#include "ReactionNetwork.hpp"
 
int main(int argc, char *argv[]) {
    CLI::App app{"CuCTMC -- Parallel tools for analyzing Chemical Master Equation CTMCs"};

    std::filesystem::path inputFile;
    app.add_option("input", inputFile, "Specify the reaction network JSON file")
        ->required();

    int warps = 1;
    app.add_option("-w,--warps", warps, "Number of sample path warps (32 paths per warp)")
        ->required()
        ->check(CLI::NonNegativeNumber);

    double tMax = 10;
    app.add_option("-T,--tMax", tMax, "Maximum simulation time")
        ->required()
        ->check(CLI::NonNegativeNumber);

    std::vector<std::tuple<std::string, std::vector<double>>> rates;
    app.add_option("--rate", rates, "Override a reaction's rate. Ex. \n--rate GammaV 15 \n--rate GammaV $(seq 1.5 1.5 150)");

    std::vector<std::tuple<std::string, std::vector<int>>> reacts;
    app.add_option("--reactants", reacts, "Override a reaction's reactants. Ex. \n--reactants Gamma 1 0 1 \n--reactants Gamma $(printf '\%i 0 1' $(seq 1 10))");

    std::vector<std::tuple<std::string, std::vector<int>>> prods;
    app.add_option("--products", prods, "Override a reaction's products. Ex. \n--reactants aI 0 35 0 1 \n--reactants aI $(printf '0 \%i 0 1' $(seq 1 50))");

    std::vector<int> conds;
    app.add_option("--ic", conds, "Override the initial conditions. Ex. \n--ic 1000 3 0 0 \n--ic $(printf '1000 \%i 0 0' $(seq 1 30))")
        ->check(CLI::NonNegativeNumber);

    std::filesystem::path outputDir{};
    auto outputDirOpt = app.add_option("-o,--output", outputDir, "Directory to place simulation output files");

    double tGrid = 0;
    auto tGridOpt = app.add_option("-t,--tGrid", tGrid, "Time grid spacing for saved trajectories")
        ->check(CLI::NonNegativeNumber);

    int saved = 0;
    app.add_option("-s,--save", saved, "Number of paths to save trajectories for")
        ->needs(tGridOpt)
        ->needs(outputDirOpt)
        ->check(CLI::NonNegativeNumber);

    CLI11_PARSE(app, argc, argv);

    ReactionNetwork ctmc(inputFile);

    //  check that all sweeps are input correctly

    std::vector<int> sweepSizes;
    int reactantsSize = ctmc.reactants.size();
    for (size_t i = 0; i < rates.size(); i++) {
       sweepSizes.push_back(std::get<1>(rates[i]).size());
    }
    for (size_t i = 0; i < reacts.size(); i++) {
        int sweepSize = std::get<1>(reacts[i]).size();
        if (sweepSize % reactantsSize != 0) {
            throw std::runtime_error("Length of required reactants must be a multiple of the number of reactants");
        }
        sweepSizes.push_back(sweepSize / reactantsSize);
    }
    for (size_t i = 0; i < prods.size(); i++) {
        int sweepSize = std::get<1>(prods[i]).size();
        if (sweepSize % reactantsSize != 0) {
            throw std::runtime_error("Length of reaction products must be a multiple of the number of reactants");
        }
        sweepSizes.push_back(sweepSize / reactantsSize);
    }
    if (!conds.empty()) {
        int sweepSize = conds.size();
        if (sweepSize % reactantsSize != 0) {
            throw std::runtime_error("Length of initial conditions must be a multiple of the number of reactants");
        }
        sweepSizes.push_back(sweepSize / reactantsSize);
    }
    if (sweepSizes.size() > 1 && std::adjacent_find(sweepSizes.begin(), sweepSizes.end(), std::not_equal_to<>()) != sweepSizes.end()) {
        throw std::runtime_error("Length of each sweep must be equal");
    }

    int sweepSize = 1;
    if (!sweepSizes.empty()) {
        sweepSize = sweepSizes.front();
    }

    //  creat sim info struct

    SSASimInfo simInfo;
    simInfo.tMax = tMax;
    simInfo.warps = warps;
    simInfo.tGrid = tGrid;
    simInfo.savedPaths = saved;
    simInfo.outputDir = outputDir; 
    simInfo.base = ctmc;

    //  create sys info structs

    std::vector<SSASysInfo> sysInfos(sweepSize);
    for (size_t i = 0; i < sweepSize; i++) {
        sysInfos[i].reactionRates = ctmc.reactionRates;
        sysInfos[i].initialConditions = ctmc.initialConditions;
        sysInfos[i].reactantCoefficients = ctmc.reactantCoefficients;
        sysInfos[i].transitionCoefficients = ctmc.transitionCoefficients;

        for (size_t j = 0; j < rates.size(); j++) {
            size_t reaction = ctmc.getReactionIdx(std::get<0>(rates[j]));
            double rate = std::get<1>(rates[j])[i];

            sysInfos[i].reactionRates[reaction] = rate;
        }

        for (size_t j = 0; j < reacts.size(); j++) {    //  reacts loop must come before prods
            size_t reaction = ctmc.getReactionIdx(std::get<0>(reacts[j]));
            std::vector<int>& react = std::get<1>(reacts[j]); 

            for (size_t k = 0; k < reactantsSize; k++) {
                size_t l = reaction * reactantsSize + k;
                size_t reactIdx = i * reactantsSize + k;

                int beta = ctmc.reactantCoefficients[l] + ctmc.transitionCoefficients[l];

                sysInfos[i].reactantCoefficients[l] = react[reactIdx];
                sysInfos[i].transitionCoefficients[l] = beta - react[reactIdx];
            }
        }

        for (size_t j = 0; j < prods.size(); j++) {
            size_t reaction = ctmc.getReactionIdx(std::get<0>(prods[j]));
            std::vector<int>& prod = std::get<1>(prods[j]);

            for (size_t k = 0; k < reactantsSize; k++) {
                size_t l = reaction * reactantsSize + k;
                size_t prodIdx = i * reactantsSize + k;

                sysInfos[i].transitionCoefficients[l] = prod[prodIdx] - sysInfos[i].reactantCoefficients[l];
            }
        }

        if (!conds.empty()) {
            for (size_t j = 0; j < reactantsSize; j++) {
                sysInfos[i].initialConditions[j] = conds[i * reactantsSize + j];
            }
        }
    }

    SSA(simInfo, sysInfos);

    return 0;
}
    