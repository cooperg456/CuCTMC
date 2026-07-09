#include "cuda.cuh"

#include "cuda_runtime.h"
#include "curand_kernel.h"

__global__
void cuSSA(int reactants, int reactions, int paths, int steps, int* d_alpha, 
           int* d_trans, double* d_rates, int* d_conds, int* d_paths, double* d_times) {
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

dataOut stochasticSimulation(int* d_alpha, int* d_trans, double* d_rates, 
                             int* initialConditions, int reactants, 
                             int reactions, int paths, int steps) {
    // get device
    int device;
    cudaGetDevice(&device);

    cudaMemLocation memloc;
    memloc.id = device;
    memloc.type = cudaMemLocationTypeDevice;

    //  calculate mem requirements
    int n_paths = paths * (steps + 1) * reactants;
    int n_times = paths * (steps + 1);
    int n_alpha = reactions * reactants;

    //  malloc path and ic buffers
    int *d_paths;
    int *d_conds;
    double *d_times;
    cudaMallocManaged(&d_paths, n_paths * sizeof(int));
    cudaMallocManaged(&d_times, n_times * sizeof(double));
    cudaMallocManaged(&d_conds, reactants * sizeof(int));

    memcpy(d_conds, initialConditions, reactants * sizeof(int));

    //  prefetches
    cudaMemPrefetchAsync(d_alpha, n_alpha * sizeof(int), memloc, 0);
    cudaMemPrefetchAsync(d_trans, n_alpha * sizeof(int), memloc, 0);
    cudaMemPrefetchAsync(d_conds, reactants * sizeof(int), memloc, 0);
    cudaMemPrefetchAsync(d_rates, reactions * sizeof(double), memloc, 0);
    cudaMemPrefetchAsync(d_paths, n_paths * sizeof(int), memloc, 0);
    cudaMemPrefetchAsync(d_times, n_times * sizeof(double), memloc, 0);

    //  kernel launch
    int blockSize = 256;
    int numBlocks = (paths + blockSize - 1) / blockSize;
    cuSSA<<<numBlocks, blockSize>>>(reactants, reactions, paths, steps, d_alpha, d_trans, d_rates, d_conds, d_paths, d_times);

    //  device sync
    cudaDeviceSynchronize();

    //  get data
    dataOut output{};
    output.paths.resize(paths, std::vector<std::vector<int>>(steps + 1, std::vector<int>(reactants)));
    output.times.resize(paths, std::vector<double>(steps + 1));

    for (size_t path = 0; path < paths; path++) {
        for (size_t step = 0; step < (steps + 1); step++) {
            memcpy(output.paths[path][step].data(),
                d_paths + (path * (steps + 1) * reactants) + (step * reactants),
                reactants * sizeof(int));
        }
        memcpy(output.times[path].data(), d_times + (path * (steps + 1)), (steps + 1) * sizeof(double));
    }

    //  free paths buffer
    cudaFree(d_paths);
    cudaFree(d_times);
    cudaFree(d_conds);

    //  return
    return output;
}
