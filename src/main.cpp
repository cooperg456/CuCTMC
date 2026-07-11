#include <iostream>
#include <fstream>

#include "CLI/CLI11.hpp"

#include "StochasticSimulation.cuh"
#include "ReactionNetwork.hpp"
 
int main(int argc, char *argv[]) {
    CLI::App app{"CuCTMC"};

    std::string networkFile;
    app.add_option("input", networkFile, "Reaction network JSON file")->required();

    SSASimInfo info{};

    app.add_option("--samplePaths", info.samplePaths, "Number of sample paths");
    app.add_option("--savedPaths", info.savedPaths, "Number of paths to save trajectories for");
    app.add_option("--tGrid", info.tGrid, "Time grid spacing for saved trajectories");
    app.add_option("--tMax", info.tMax, "Maximum simulation time");
    app.add_option("--ic", info.initialConditions, "Initial conditions, e.g. --ic 1000 3 0 0");

    std::vector<std::string> rateOverrides;
    app.add_option("--rate", rateOverrides, "Override a reaction rate by name, e.g. --rate '/gamma_V=15.0'");

    CLI11_PARSE(app, argc, argv);

    ReactionNetwork network(networkFile);

    for (auto& kv : rateOverrides) {
        auto pos = kv.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("bad --rate format, expected name=value: " + kv);
        }

        std::string name = kv.substr(0, pos);
        double value = std::stod(kv.substr(pos + 1));

        auto it = std::find(network.reactions.begin(), network.reactions.end(), name);
        if (it == network.reactions.end()) {
            throw std::runtime_error("unknown reaction name: " + name);
        }

        size_t idx = std::distance(network.reactions.begin(), it);
        network.reactionRates[idx] = value;
    }

    SSASimOutput out = stochasticSimulation(network, info);

    return 0;
}
