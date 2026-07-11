#include "StochasticSimulation.cuh"

#include <iostream>
#include <fstream>

#include "cuda_runtime.h"
#include "curand_kernel.h"

__global__
void cuSSA(int reactants, int reactions, int paths, int savedPaths,
           double tGrid, double tMax, double* d_rates, int* d_alpha, 
           int* d_trans, int* d_conds,  double* d_times, int* d_paths) {
    //  thread indexing

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    int pointsPerPath = tMax / tGrid + 1;

    for (int path = index; path < paths; path += stride) {
        //  initialize cuRAND

        curandStatePhilox4_32_10 state;
        curand_init(9384, path, 0, &state);

        //  Initialize t = t0 and x = x0, set n = 0.

        int x[MAX_SSA_REACTANTS];
        for (int i = 0; i < reactants; i++) {
            x[i] = d_conds[i];
        }
        double t = 0;

        //  record first n paths

        int tSaved = 0;
        if (path < savedPaths) {
            for (int i = 0; i < reactants; i++) {
                d_paths[(path * pointsPerPath * reactants) + i] = x[i];
            }
            d_times[path * pointsPerPath] = t;
        }

        //  while (t < tMax);

        do {
            //  Compute a_j(x), j = 1, 2, ..., K and a_0(x).

            double a[MAX_SSA_REACTIONS];
            double a0 = 0;
            for (int j = 0; j < reactions; j++) {
                a[j] = d_rates[j];
                for (int m = 0; m < reactants; m++) {
                    int alpha_jm = d_alpha[j * reactants + m];
                    if (alpha_jm > 0) {
                        if (x[m] < alpha_jm) {
                            a[j] = 0.0;
                            break;
                        }
                        for (int k = x[m] - alpha_jm + 1; k <= x[m]; k++) {
                            a[j] *= k;
                        }
                    }
                }
                a0 += a[j];
            }

            if (a0 != 0) {
                //  Generate r_1, r_2 ∼ U([0, 1]).

                double2 r = curand_uniform2_double(&state);

                //  Use the Golden rule to transform r_1 into τ ∼ Exp(a_0(x))

                double tau = log(1 / r.x) / a0;   

                //  Let j be the smallest integer for which

                int j = 0;
                double sum_a = a[0];
                for (; sum_a < r.y * a0 && j < reactions - 1;) {
                    j++;
                    sum_a += a[j];
                }

                //  Increment t by τ and x by v_j

                for (int i = 0; i < reactants; i++) {
                    x[i] += d_trans[reactants * j + i];
                }
                t += tau;
            }
            else {
                t = tMax;
            }

            //  record first n paths

            if (path < savedPaths) {
                while (tSaved * tGrid < t && tSaved < pointsPerPath - 1) {
                    tSaved++;
                    
                    for (int i = 0; i < reactants; i++) {
                        d_paths[(path * pointsPerPath * reactants) + (tSaved * reactants) + i] = x[i];
                    }
                    d_times[path * pointsPerPath + tSaved] = tSaved * tGrid;
                }
            }
        } while (t < tMax);
    }
}

SSASimOutput stochasticSimulation(ReactionNetwork &network, SSASimInfo &info) {
    //  ReactionNetwork info

    double *d_reactionRates;
    int *d_reactantCoefficients;
    int *d_transitionCoefficients;

    int n_reactionRates = network.reactions.size();
    int n_reactantCoefficients = network.reactantCoefficients.size();
    int n_transitionCoefficients = network.transitionCoefficients.size();

    //  SSASimInfo info

    double *d_timePoints;
    int *d_initialConditions;
    int *d_samplePaths;
    
    int n_timePoints = info.savedPaths * (info.tMax / info.tGrid + 1);
    int n_initialConditions = info.initialConditions.size();
    int n_samplePaths = n_timePoints * n_initialConditions;
    
    //  malloc buffers
    
    cudaMallocManaged(&d_reactionRates, n_reactionRates * sizeof(double));
    cudaMallocManaged(&d_reactantCoefficients, n_reactantCoefficients * sizeof(int));
    cudaMallocManaged(&d_transitionCoefficients, n_transitionCoefficients * sizeof(int));
    cudaMallocManaged(&d_initialConditions, n_initialConditions * sizeof(int));

    cudaMallocManaged(&d_timePoints, n_timePoints * sizeof(double));
    cudaMallocManaged(&d_samplePaths, n_samplePaths * sizeof(int));

    //  memcpy to gpu

    memcpy(d_reactionRates, network.reactionRates.data(), n_reactionRates * sizeof(double));
    memcpy(d_reactantCoefficients, network.reactantCoefficients.data(), n_reactantCoefficients * sizeof(int));
    memcpy(d_transitionCoefficients, network.transitionCoefficients.data(), n_transitionCoefficients * sizeof(int));
    memcpy(d_initialConditions, info.initialConditions.data(), n_initialConditions * sizeof(int));
    
    //  prefetches

    int device;
    cudaGetDevice(&device);

    cudaMemLocation memlocDev;
    memlocDev.id = device;
    memlocDev.type = cudaMemLocationTypeDevice;

    cudaMemPrefetchAsync(d_reactionRates, n_reactionRates * sizeof(double), memlocDev, 0);
    cudaMemPrefetchAsync(d_reactantCoefficients, n_reactantCoefficients * sizeof(int), memlocDev, 0);
    cudaMemPrefetchAsync(d_transitionCoefficients, n_transitionCoefficients * sizeof(int), memlocDev, 0);
    cudaMemPrefetchAsync(d_initialConditions, n_initialConditions * sizeof(int), memlocDev, 0);

    cudaMemPrefetchAsync(d_timePoints, n_timePoints * sizeof(double), memlocDev, 0);
    cudaMemPrefetchAsync(d_samplePaths, n_samplePaths * sizeof(int), memlocDev, 0);
    
    //  kernel launch
    
    int blockSize = 256;
    int numBlocks = (info.samplePaths + blockSize - 1) / blockSize;
    cuSSA<<<numBlocks, blockSize>>>(network.reactants.size(), network.reactions.size(), info.samplePaths, 
                                    info.savedPaths, info.tGrid, info.tMax, d_reactionRates, d_reactantCoefficients, 
                                    d_transitionCoefficients, d_initialConditions, d_timePoints, d_samplePaths);
                                
    cudaDeviceSynchronize();

    //  free allocations

    cudaFree(d_reactionRates);
    cudaFree(d_reactantCoefficients);
    cudaFree(d_transitionCoefficients);
    cudaFree(d_initialConditions);

    return SSASimOutput(d_timePoints, d_samplePaths, network, info);
}

SSASimOutput::SSASimOutput(double *timePoints, int *samplePaths, ReactionNetwork &network, SSASimInfo &simInfo) 
                           : d_timePoints(timePoints), d_samplePaths(samplePaths), network(network), info(simInfo) {}

SSASimOutput::~SSASimOutput() {
    cudaFree(d_timePoints);
    cudaFree(d_samplePaths);
}

void SSASimOutput::toCSV(const std::filesystem::path& csvFile) const {
    int pointsPerPath = info.tMax / info.tGrid + 1;
    int n_timePoints = info.savedPaths * pointsPerPath;
    int n_initialConditions = info.initialConditions.size();
    int n_samplePaths = n_timePoints * n_initialConditions;
    
    //  more prefetches

    cudaMemLocation memlocHost;
    memlocHost.id = 0;
    memlocHost.type = cudaMemLocationTypeHost;
    
    cudaMemPrefetchAsync(d_timePoints, n_timePoints * sizeof(double), memlocHost, 0);
    cudaMemPrefetchAsync(d_samplePaths, n_samplePaths * sizeof(int), memlocHost, 0);

    cudaDeviceSynchronize();

    //  get data
    
    std::vector<std::vector<std::vector<int>>> paths(info.savedPaths, std::vector<std::vector<int>>(pointsPerPath, std::vector<int>(network.reactants.size())));
    std::vector<std::vector<double>> times(info.savedPaths, std::vector<double>(pointsPerPath));

    for (size_t path = 0; path < info.savedPaths; path++) {
        for (size_t step = 0; step < pointsPerPath; step++) {
            memcpy(paths[path][step].data(),
                d_samplePaths + (path * pointsPerPath * network.reactants.size()) + (step * network.reactants.size()),
                network.reactants.size() * sizeof(int));
        }
        memcpy(times[path].data(), d_timePoints + (path * pointsPerPath), pointsPerPath * sizeof(double));
    }

    //  write data

    std::ofstream file(csvFile);
    if (file.is_open()) {
        size_t reactants = info.initialConditions.size();

        file << "path,time";
        for (size_t r = 0; r < reactants; r++) {
            file << ",reactant_" << r;
        }
        file << "\n";

        for (size_t path = 0; path < info.savedPaths; path++) {
            for (size_t step = 0; step < paths[path].size(); step++) {
                file << path << "," << times[path][step];
                for (size_t r = 0; r < reactants; r++) {
                    file << "," << paths[path][step][r];
                }
                file << "\n";
            }
        }
        file.close();
    } else {
        std::cerr << "Failed to open csv file for writing\n";
    }
}