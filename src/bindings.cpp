#include <pybind11/pybind11.h>
#include <cstdint>

namespace py = pybind11;

// Declare the external C-functions exactly as they appear in cuda_solver.cu
extern "C" {
    void launch_spmv(uintptr_t K0_ptr, uintptr_t rho_ptr, uintptr_t dofs_ptr, uintptr_t p_ptr, uintptr_t q_ptr, int num_elements, int num_dofs, double dynamic_penalty);
    void launch_spmv_2d(uintptr_t K0_ptr, uintptr_t rho_ptr, uintptr_t dofs_ptr, uintptr_t p_ptr, uintptr_t q_ptr, int num_elements, int num_dofs, double dynamic_penalty);
    void launch_spmv_2d_subwarp_sym(uintptr_t K0_ptr, uintptr_t rho_ptr, uintptr_t dofs_ptr, uintptr_t p_ptr, uintptr_t q_ptr, int num_elements, int num_dofs, double dynamic_penalty);
}

PYBIND11_MODULE(pravaha_cu_engine, m) {
    m.doc() = "PRAVAHA V2 Native C++/CUDA Zero-Copy Solver";
    
    m.def("launch_spmv", &launch_spmv, "3D Matrix-Free SpMV Kernel");
    m.def("launch_spmv_2d", &launch_spmv_2d, "2D Baseline Matrix-Free SpMV Kernel");
    m.def("launch_spmv_2d_subwarp_sym", &launch_spmv_2d_subwarp_sym, "2D Symmetric Matrix-Free SpMV Kernel");
}