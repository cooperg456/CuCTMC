#pragma once

#include <vector>

class ChemicalMasterEquation {
public:
    ChemicalMasterEquation(std::vector<std::vector<int>>&,
                           std::vector<std::vector<int>>&,
                           std::vector<double>&);
    ~ChemicalMasterEquation();

    const std::vector<std::vector<int>> alpha;
    const std::vector<std::vector<int>> beta;
    const std::vector<double> rates;

    void allocateDeviceResources();
    void freeDeviceResources();

    int* get_d_alpha() const;
    int* get_d_trans() const;
    double* get_d_rates() const;

private:
    int *d_alpha = nullptr;
    int *d_trans = nullptr;
    double *d_rates = nullptr;
};
