#include "ReactionNetwork.hpp"

#include <iostream>
#include <fstream>

#include "nlohmann/json.hpp"

ReactionNetwork::ReactionNetwork(const std::filesystem::path& jsonFile) {
    std::ifstream f(jsonFile);

    if (!f.is_open()) {
        std::string msg = "Failed to open file: " + jsonFile.string();
        if (f.bad()) msg += " (fatal stream error)";
        if (f.fail()) msg += std::string(" - ") + std::strerror(errno);

        f.close();
        throw std::runtime_error(msg);
    }

    nlohmann::json network = nlohmann::json::parse(f);
    f.close();

    size_t n_reactants = network["reactants"].size();
    size_t n_reactions = network["reactions"].size();

    reactants = network["reactants"];

    reactions.resize(n_reactions);
    reactionRates.resize(n_reactions);
    reactantCoefficients = std::vector<int>(n_reactants * n_reactions, 0);
    transitionCoefficients = std::vector<int>(n_reactants * n_reactions, 0);

    for (size_t i = 0; i < network["reactions"].size(); i++) {
        reactions[i] = network["reactions"][i]["name"];
        reactionRates[i] = network["reactions"][i]["rate"];

        for (auto& reactant : network["reactions"][i]["reactants"].items()) {
            auto it = std::find(reactants.begin(), reactants.end(), reactant.key());
            size_t j = std::distance(reactants.begin(), it);
            
            reactantCoefficients[i * n_reactants + j] += (int)reactant.value();
            transitionCoefficients[i * n_reactants + j] -= (int)reactant.value();
        }

        for (auto& product : network["reactions"][i]["products"].items()) {
            auto it = std::find(reactants.begin(), reactants.end(), product.key());
            size_t j = std::distance(reactants.begin(), it);
            
            transitionCoefficients[i * n_reactants + j] += (int)product.value();
        }
    }
}
