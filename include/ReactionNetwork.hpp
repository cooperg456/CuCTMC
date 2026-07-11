#pragma once

#include <vector>
#include <string>
#include <filesystem>

class ReactionNetwork {
public:
    ReactionNetwork() = default;
    explicit ReactionNetwork(const std::filesystem::path& jsonFile);

    //  reaction network parameters
    
    std::vector<std::string> reactants{};
    std::vector<std::string> reactions{};

    std::vector<double> reactionRates{};
    std::vector<int> reactantCoefficients{};
    std::vector<int> transitionCoefficients{};
};
