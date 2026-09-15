#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// =============================================================================
// CONSTANTS & LOOKUP TABLES
// =============================================================================

// Maps (row * 6 + col) -> [0..20] upper triangle index for symmetric 6x6 matvec
__constant__ int sym_lut_6x6[36] = {
     0,  1,  2,  3,  4,  5,   // Row 0
     1,  6,  7,  8,  9, 10,   // Row 1
     2,  7, 11, 12, 13, 14,   // Row 2
     3,  8, 12, 15, 16, 17,   // Row 3
     4,  9, 13, 16, 18, 19,   // Row 4
     5, 10, 14, 17, 19, 20    // Row 5
};

// Maps (row * 12 + col) -> [0..77] upper triangle index for symmetric 12x12 matvec
__constant__ int sym_lut_12x12[144] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
     1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
     2, 13, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
     3, 14, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41,
     4, 15, 25, 34, 42, 43, 44, 45, 46, 47, 48, 49,
     5, 16, 26, 35, 43, 50, 51, 52, 53, 54, 55, 56,
     6, 17, 27, 36, 44, 51, 57, 58, 59, 60, 61, 62,
     7, 18, 28, 37, 45, 52, 58, 63, 64, 65, 66, 67,
     8, 19, 29, 38, 46, 53, 59, 64, 68, 69, 70, 71,
     9, 20, 30, 39, 47, 54, 60, 65, 69, 72, 73, 74,
    10, 21, 31, 40, 48, 55, 61, 66, 70, 73, 75, 76,
    11, 22, 32, 41, 49, 56, 62, 67, 71, 74, 76, 77
};

// =============================================================================
// KERNEL 1: 3D Tet4 Matrix-Free SpMV (1 Thread / Element, 12x12 SoA)
// =============================================================================
__global__ void spmv_kernel_3d(const double* __restrict__ K0, 
                               const double* __restrict__ rho, 
                               const int* __restrict__ dofs, 
                               const double* __restrict__ p, 
                               double* __restrict__ q, 
                               int num_elements,
                               double dynamic_penalty) { 
                                
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;

    // 1. SIMP Penalization
    double r = rho[e];
    double penalty = exp(dynamic_penalty * log(r)); 

    // 2. Gather displacements using SoA memory coalescing
    double p_local[12];
    int d[12];
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        d[i] = dofs[i * num_elements + e];
        p_local[i] = p[d[i]];
    }

    // 3. 12x12 Dense Matrix-Vector Multiply with SoA layout
    double q_local[12] = {0.0};
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        double sum = 0.0;
        #pragma unroll
        for(int j = 0; j < 12; j++) {
            sum += K0[(i * 12 + j) * num_elements + e] * p_local[j];
        }
        q_local[i] = sum * penalty;
    }

    // 4. Atomic scatter to global vector
    #pragma unroll
    for(int i = 0; i < 12; i++) {
        atomicAdd(&q[d[i]], q_local[i]);
    }
}

// =============================================================================
// KERNEL 2: 2D Tri3 Baseline SpMV (Step 1: 1 Thread / Element, Full 6x6 SoA)
// =============================================================================
__global__ void spmv_kernel_2d(const double* __restrict__ K0, 
                               const double* __restrict__ rho, 
                               const int* __restrict__ dofs, 
                               const double* __restrict__ p, 
                               double* __restrict__ q, 
                               int num_elements,
                               double dynamic_penalty) { 
                                   
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;

    double r = rho[e];
    double penalty = exp(dynamic_penalty * log(r)); 

    double p_local[6];
    int d[6];
    #pragma unroll
    for(int i = 0; i < 6; i++) {
        d[i] = dofs[i * num_elements + e];
        p_local[i] = p[d[i]];
    }

    double q_local[6] = {0.0};
    #pragma unroll
    for(int i = 0; i < 6; i++) {
        double sum = 0.0;
        #pragma unroll
        for(int j = 0; j < 6; j++) {
            sum += K0[(i * 6 + j) * num_elements + e] * p_local[j];
        }
        q_local[i] = sum * penalty;
    }

    #pragma unroll
    for(int i = 0; i < 6; i++) {
        atomicAdd(&q[d[i]], q_local[i]);
    }
}

// =============================================================================
// KERNEL 3: 2D Tri3 Sub-Warp Symmetric SpMV (Step 4: 8 Threads / Element, 21 SoA)
// =============================================================================
__global__ void spmv_kernel_2d_subwarp_sym(const double* __restrict__ K0_sym, 
                                           const double* __restrict__ rho, 
                                           const int* __restrict__ dofs, 
                                           const double* __restrict__ p, 
                                           double* __restrict__ q, 
                                           int num_elements,
                                           double dynamic_penalty) {
    
    int global_tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id   = global_tid >> 3;         // 8 threads per sub-warp
    int lane         = threadIdx.x & 7;         // Local lane in sub-warp (0..7)
    int lane_in_warp = threadIdx.x & 31;        // Lane in 32-thread warp (0..31)

    if (subwarp_id >= num_elements) return;

    // Sub-warp mask for register shuffles (8 lanes aligned to 8-thread boundary)
    unsigned int subwarp_mask = 0xFFu << (lane_in_warp & 24);

    // 1. Calculate SIMP penalty once per element and broadcast
    double penalty = 1.0;
    if (lane == 0) {
        double r = rho[subwarp_id];
        penalty = exp(dynamic_penalty * log(r));
    }
    penalty = __shfl_sync(subwarp_mask, penalty, (lane_in_warp & 24));

    // 2. Cooperative gather: lanes 0..5 load their respective DOF & displacement
    int my_dof = 0;
    double my_p = 0.0;
    if (lane < 6) {
        my_dof = dofs[lane * num_elements + subwarp_id];
        my_p   = p[my_dof];
    }

    // 3. Dot product via warp shuffles (all 8 threads participate in __shfl_sync)
    double row_sum = 0.0;
    int row_offset = lane * 6;

    #pragma unroll
    for (int j = 0; j < 6; j++) {
        double pj = __shfl_sync(subwarp_mask, my_p, (lane_in_warp & 24) + j);
        if (lane < 6) {
            int sym_idx = sym_lut_6x6[row_offset + j];
            row_sum += K0_sym[sym_idx * num_elements + subwarp_id] * pj;
        }
    }

    // 4. Atomic scatter from active lanes (0..5)
    if (lane < 6) {
        atomicAdd(&q[my_dof], row_sum * penalty);
    }
}

// =============================================================================
// HOST LAUNCHERS (Zero-Copy Pointers)
// =============================================================================
extern "C" {

void launch_spmv(uintptr_t K0_ptr,
                 uintptr_t rho_ptr,
                 uintptr_t dofs_ptr,
                 uintptr_t p_ptr,
                 uintptr_t q_ptr,
                 int num_elements,
                 int num_dofs,
                 double dynamic_penalty) {

    const double* d_K0   = reinterpret_cast<const double*>(K0_ptr);
    const double* d_rho  = reinterpret_cast<const double*>(rho_ptr);
    const int*    d_dofs = reinterpret_cast<const int*>(dofs_ptr);
    const double* d_p    = reinterpret_cast<const double*>(p_ptr);
    double*       d_q    = reinterpret_cast<double*>(q_ptr);

    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    spmv_kernel_3d<<<grid_size, block_size>>>(
        d_K0, d_rho, d_dofs, d_p, d_q, num_elements, dynamic_penalty
    );
}

void launch_spmv_2d(uintptr_t K0_ptr,
                    uintptr_t rho_ptr,
                    uintptr_t dofs_ptr,
                    uintptr_t p_ptr,
                    uintptr_t q_ptr,
                    int num_elements,
                    int num_dofs,
                    double dynamic_penalty) {

    const double* d_K0   = reinterpret_cast<const double*>(K0_ptr);
    const double* d_rho  = reinterpret_cast<const double*>(rho_ptr);
    const int*    d_dofs = reinterpret_cast<const int*>(dofs_ptr);
    const double* d_p    = reinterpret_cast<const double*>(p_ptr);
    double*       d_q    = reinterpret_cast<double*>(q_ptr);

    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    spmv_kernel_2d<<<grid_size, block_size>>>(
        d_K0, d_rho, d_dofs, d_p, d_q, num_elements, dynamic_penalty
    );
}

void launch_spmv_2d_subwarp_sym(uintptr_t K0_sym_ptr,
                                uintptr_t rho_ptr,
                                uintptr_t dofs_ptr,
                                uintptr_t p_ptr,
                                uintptr_t q_ptr,
                                int num_elements,
                                int num_dofs,
                                double dynamic_penalty) {

    const double* d_K0_sym = reinterpret_cast<const double*>(K0_sym_ptr);
    const double* d_rho    = reinterpret_cast<const double*>(rho_ptr);
    const int*    d_dofs   = reinterpret_cast<const int*>(dofs_ptr);
    const double* d_p      = reinterpret_cast<const double*>(p_ptr);
    double*       d_q      = reinterpret_cast<double*>(q_ptr);

    int block_size = 256;
    int total_threads = num_elements * 8;
    int grid_size = (total_threads + block_size - 1) / block_size;

    spmv_kernel_2d_subwarp_sym<<<grid_size, block_size>>>(
        d_K0_sym, d_rho, d_dofs, d_p, d_q, num_elements, dynamic_penalty
    );
}

} // extern "C"

