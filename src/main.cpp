#include <iostream>
#include <vector>

#include "cuda.cuh"
 
int main(void)
{
    int N = 1<<20;
    
    std::vector<float> x(N, 1.0f);
    std::vector<float> y(N, 2.0f);
    
    add(x, y);
    
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = std::max(maxError, std::abs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    
    return 0;
}
