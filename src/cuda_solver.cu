#include <cuda_runtime.h>
#include <cstdint>

// The Fused Matrix-Free SpMV Kernel
__global__ void spmv_kernel(const double* __restrict__ K0, 
                            const double* __restrict__ rho, 
                            const int* __restrict__ dofs, 
                            const double* __restrict__ p, 
                            double* __restrict__ q, 
                            int num_elements,
                            double dynamic_penalty) { 
                                
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;

    // 1. Calculate the SIMP Penalty using Fast Math (Density is strictly > 0)
    double r = rho[e];
    double penalty = exp(dynamic_penalty * log(r)); 

    // 2. Gather displacements using SoA (Structure-of-Arrays) memory coalescing
    double p_local[12];
    int d[12];
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        // SoA Indexing ensures 32 threads in a warp read perfectly adjacent memory addresses
        d[i] = dofs[i * num_elements + e];
        p_local[i] = p[d[i]];
    }

    // 3. Perform the 12x12 Dense Matrix Multiplication with SoA indexing
    double q_local[12] = {0.0};
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        double sum = 0.0;
        #pragma unroll
        for(int j = 0; j < 12; j++) {
            // SoA Index: ((Row * 12) + Col) * Num_Elements + Element_Index
            sum += K0[(i * 12 + j) * num_elements + e] * p_local[j];
        }
        q_local[i] = sum * penalty;
    }

    // 4. Safely scatter the calculated local forces
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        atomicAdd(&q[d[i]], q_local[i]);
    }
}