#include "cuda.cuh"

#include "cuda_runtime.h"
#include "curand_kernel.h"

__global__
void cuSSA(int reactants, int reactions, int paths, 
    int steps, double* d_rates, int* d_alpha, int* d_trans, 
    int* d_conds,  double* d_times, int* d_paths) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int path = index; path < paths; path += stride) {
            //  initialize cuRAND
            curandStatePhilox4_32_10 state;
            curand_init(9384, path, 0, &state);

            //  Initialize t = t0 and x = x0, set n = 0.
            int x[MAX_REACTANTS];
            for (int i = 0; i < reactants; i++) {
                x[i] = d_conds[i];
            }
            double t = 0;

            for (int i = 0; i < reactants; i++) {
                d_paths[(path * (steps + 1) * reactants) + i] = x[i];
            }
            d_times[path * (steps + 1)] = t;

        for (int n = 0; n < steps; n++) {
            //  Compute a_j(x), j = 1, 2, ..., K and a_0(x).
            double a[MAX_REACTIONS];
            double a0 = 0;
            for (int j = 0; j < reactions; j++) {
                a[j] = d_rates[j];
            
                for (int m = 0; m < reactants; m++) {
                    for (int k = x[m] - d_alpha[j * reactants + m]; k < x[m]; k++) {
                        a[j] *= k;
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

            //  record data
            for (int i = 0; i < reactants; i++) {
                d_paths[(path * (steps + 1) * reactants) + ((n + 1) * reactants) + i] = x[i];
            }
            d_times[path * (steps + 1) + (n + 1)] = t;
        }
    }
}

dataOut stochasticSimulation(ReactionNetwork cme) {
    double *d_reactionRates;
    int *d_reactantCoefficients;
    int *d_transitionCoefficients;
    int *d_initialConditions;

    double *d_timePoints;
    int *d_samplePaths;

    int n_reactionRates = cme.reactions.size();
    int n_reactantCoefficients = cme.reactantCoefficients.size();
    int n_transitionCoefficients = cme.transitionCoefficients.size();
    int n_initialConditions = cme.reactants.size();

    int n_timePoints = cme.samplePaths * (cme.simulationSteps + 1);
    int n_samplePaths = n_timePoints * n_initialConditions;
    
    //  malloc buffers
    
    cudaMallocManaged(&d_reactionRates, n_reactionRates * sizeof(double));
    cudaMallocManaged(&d_reactantCoefficients, n_reactantCoefficients * sizeof(int));
    cudaMallocManaged(&d_transitionCoefficients, n_transitionCoefficients * sizeof(int));
    cudaMallocManaged(&d_initialConditions, n_initialConditions * sizeof(int));

    cudaMallocManaged(&d_timePoints, n_timePoints * sizeof(double));
    cudaMallocManaged(&d_samplePaths, n_samplePaths * sizeof(int));

    //  memcpy to gpu

    memcpy(d_reactionRates, cme.reactionRates.data(), n_reactionRates * sizeof(double));
    memcpy(d_reactantCoefficients, cme.reactantCoefficients.data(), n_reactantCoefficients * sizeof(int));
    memcpy(d_transitionCoefficients, cme.transitionCoefficients.data(), n_transitionCoefficients * sizeof(int));
    memcpy(d_initialConditions, cme.initialConditions.data(), n_initialConditions * sizeof(int));

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
    int numBlocks = (cme.samplePaths + blockSize - 1) / blockSize;
    cuSSA<<<numBlocks, blockSize>>>(cme.reactants.size(), cme.reactions.size(), cme.samplePaths, 
        cme.simulationSteps, d_reactionRates, d_reactantCoefficients, d_transitionCoefficients, 
        d_initialConditions, d_timePoints, d_samplePaths);

    //  more prefetches

    cudaMemLocation memlocHost;
    memlocHost.id = 0;
    memlocHost.type = cudaMemLocationTypeHost;
    
    cudaMemPrefetchAsync(d_timePoints, n_timePoints * sizeof(double), memlocHost, 0);
    cudaMemPrefetchAsync(d_samplePaths, n_samplePaths * sizeof(int), memlocHost, 0);

    cudaDeviceSynchronize();

    //  get data

    int paths = cme.samplePaths;
    int steps = cme.simulationSteps;
    int reactions = cme.reactions.size();
    int reactants = cme.reactants.size();
    
    dataOut output{};
    output.paths.resize(paths, std::vector<std::vector<int>>(steps + 1, std::vector<int>(reactants)));
    output.times.resize(paths, std::vector<double>(steps + 1));

    for (size_t path = 0; path < paths; path++) {
        for (size_t step = 0; step < (steps + 1); step++) {
            memcpy(output.paths[path][step].data(),
                d_samplePaths + (path * (steps + 1) * reactants) + (step * reactants),
                reactants * sizeof(int));
        }
        memcpy(output.times[path].data(), d_timePoints + (path * (steps + 1)), (steps + 1) * sizeof(double));
    }

    //  free paths buffer

    cudaFree(d_reactionRates);
    cudaFree(d_reactantCoefficients);
    cudaFree(d_transitionCoefficients);
    cudaFree(d_initialConditions);

    cudaFree(d_timePoints);
    cudaFree(d_samplePaths);

    //  return

    return output;
}
