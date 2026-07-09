#include "cme.hpp"

#include <cstring>

#include "cuda_runtime.h"

ChemicalMasterEquation::ChemicalMasterEquation(std::vector<std::vector<int>>& alpha,
                                               std::vector<std::vector<int>>& beta,
                                               std::vector<double>& rates) 
                                               : alpha(alpha), beta(beta), rates(rates) {
    allocateDeviceResources();
}

ChemicalMasterEquation::~ChemicalMasterEquation() {
    freeDeviceResources();
}

void ChemicalMasterEquation::allocateDeviceResources() {
    freeDeviceResources();

    size_t rows = alpha.size();
    size_t cols = alpha.front().size();
    size_t n_alpha = rows * cols;
    size_t n_rates = rates.size();

    cudaMallocManaged(&d_alpha, n_alpha * sizeof(int));
    cudaMallocManaged(&d_trans, n_alpha * sizeof(int));
    cudaMallocManaged(&d_rates, n_rates * sizeof(double));

    std::vector<std::vector<int>> trans(beta);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            trans[j][i] = beta[j][i] - alpha[j][i];
        }
        memcpy(d_alpha + (j * cols), alpha[j].data(), cols * sizeof(int));
        memcpy(d_trans + (j * cols), trans[j].data(), cols * sizeof(int));
    }
    memcpy(d_rates, rates.data(), n_rates * sizeof(double));
}

void ChemicalMasterEquation::freeDeviceResources() {
    if (d_alpha) {
        cudaFree(d_alpha);
    }
    if (d_trans) {
        cudaFree(d_trans);
    }
    if (d_rates) {
        cudaFree(d_rates);
    }
}

int* ChemicalMasterEquation::get_d_alpha() const {
    return d_alpha;
}

int* ChemicalMasterEquation::get_d_trans() const {
    return d_trans;
}

double* ChemicalMasterEquation::get_d_rates() const {
    return d_rates;
}