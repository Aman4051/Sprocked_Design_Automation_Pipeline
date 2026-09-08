#include <pybind11/pybind11.h>
#include <cstdint>

namespace py = pybind11;

// Forward declaration of the CUDA launcher
// --- THE FIX: Added 'double dynamic_penalty' to match cuda_solver.cu ---
extern void run_fused_spmv(uintptr_t K0_ptr, uintptr_t rho_ptr, uintptr_t dofs_ptr, uintptr_t p_ptr, uintptr_t q_ptr, int num_elements, double dynamic_penalty);

PYBIND11_MODULE(pravaha_cu_engine, m) {
    m.doc() = "PRAVAHA V2 Native C++/CUDA Zero-Copy Solver";
    m.def("launch_spmv", &run_fused_spmv, "Launches Fused SpMV using raw CuPy Device Pointers");
}