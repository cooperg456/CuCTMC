#include "cuda.cuh"

#include "cuda_runtime.h"

__global__
void cuadd(int n, float *x, float *y) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

//  parallel vector add with x[i] + y[i] -> y[i]
void add(std::vector<float> &u, std::vector<float> &v) {
    // assumes u.size() == v.size()
    int n = u.size();

    float *x, *y;
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&y, n*sizeof(float));

    memcpy(x, u.data(), n*sizeof(float));
    memcpy(y, v.data(), n*sizeof(float));

    int device;
    cudaGetDevice(&device);

    cudaMemLocation memloc;
    memloc.id = device;
    memloc.type = cudaMemLocationTypeDevice;

    cudaMemPrefetchAsync(x, n*sizeof(float), memloc, 0);
    cudaMemPrefetchAsync(y, n*sizeof(float), memloc, 0);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cuadd<<<numBlocks, blockSize>>>(n, x, y);

    cudaDeviceSynchronize();

    memcpy(v.data(), y, n*sizeof(float));

    cudaFree(x);
    cudaFree(y);
}
