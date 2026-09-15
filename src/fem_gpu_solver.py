import os
import sys
import time
import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.sparse.linalg as cplinalg

# Point directly to the CMake build folder for the CUDA Engine
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(build_dir)
import pravaha_cu_engine

class FEMSolverGPU:
    """
    PRAVAHA V2 Matrix-Free 2D GPU Computational Engine.
    Executes Plane-Stress Tri3 FEA in VRAM without CSR sparse matrix assembly.
    """
    def __init__(self, physics_payload, mesh_data, config, use_symmetric=False):
        print("   -> Initializing 2D Matrix-Free GPU Plane-Stress Solver...")
        self.payload = physics_payload
        self.mesh = mesh_data
        self.config = config
        self.use_symmetric = use_symmetric
        
        self.E0 = np.float64(self.payload['E_derated'])
        self.nu = np.float64(self.payload['nu'])
        self.thickness = float(self.config.get('geometry', {}).get('web_thickness_m', 0.007))
        self.p_penalty = float(config.get('optimization_solver', {}).get('simp_penalization_exponent', 3.0))
        self.cg_tol = float(config.get('optimization_solver', {}).get('cg_tolerance', 1e-3))
        self.cg_maxiter = int(config.get('optimization_solver', {}).get('cg_max_iterations', 2500))
        
        self.num_nodes = len(self.mesh['nodes'])
        self.num_elements = len(self.mesh['elements'])
        self.num_dofs = self.num_nodes * 2 
        
        self._map_degrees_of_freedom()
        self._precompute_tri3_kinematics()
        
        # Preallocate SpMV persistent scratch buffers to prevent VRAM churn
        self.p_global_buf = cp.zeros(self.num_dofs, dtype=cp.float64)
        self.q_global_buf = cp.zeros(self.num_dofs, dtype=cp.float64)
        
        # Warm-Start Memory
        self.U_prev_BOL = None
        self.U_prev_EOL = None

    def _map_degrees_of_freedom(self):
        """Constructs DOF mappings and extracts free boundaries."""
        fixed_dofs = []
        for n in self.mesh['fixed_nodes']:
            fixed_dofs.extend([n * 2, n * 2 + 1])
            
        all_dofs = np.arange(self.num_dofs, dtype=np.int32)
        self.free_dofs = cp.array(np.setdiff1d(all_dofs, fixed_dofs), dtype=cp.int32)
        
        # Element DOF mapping (num_elements, 6)
        elements = self.mesh['elements']
        edof = np.zeros((self.num_elements, 6), dtype=np.int32)
        for i in range(3):
            edof[:, 2 * i]     = elements[:, i] * 2
            edof[:, 2 * i + 1] = elements[:, i] * 2 + 1
            
        self.elem_dofs_gpu = cp.array(edof, dtype=cp.int32)
        # SoA transposition (6, num_elements) for coalesced warp reads
        self.elem_dofs_gpu_soa = cp.ascontiguousarray(self.elem_dofs_gpu.T)

    def _precompute_tri3_kinematics(self):
        """Vectorized 2D Plane-Stress Element Matrix Assembly directly in SoA layout."""
        c = self.E0 / (1.0 - self.nu**2)
        D = c * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0]
        ], dtype=np.float64)
        
        nodes = self.mesh['nodes']
        elements = self.mesh['elements']
        p1 = nodes[elements[:, 0]]
        p2 = nodes[elements[:, 1]]
        p3 = nodes[elements[:, 2]]
        
        b1 = p2[:, 1] - p3[:, 1]
        b2 = p3[:, 1] - p1[:, 1]
        b3 = p1[:, 1] - p2[:, 1]
        
        c1 = p3[:, 0] - p2[:, 0]
        c2 = p1[:, 0] - p3[:, 0]
        c3 = p2[:, 0] - p1[:, 0]
        
        areas = np.maximum(self.mesh['areas'], 1e-12)
        
        B = np.zeros((self.num_elements, 3, 6), dtype=np.float64)
        B[:, 0, 0] = b1; B[:, 0, 2] = b2; B[:, 0, 4] = b3
        B[:, 1, 1] = c1; B[:, 1, 3] = c2; B[:, 1, 5] = c3
        B[:, 2, 0] = c1; B[:, 2, 1] = b1; B[:, 2, 2] = c2; B[:, 2, 3] = b2; B[:, 2, 4] = c3; B[:, 2, 5] = b3
        B /= (2.0 * areas[:, np.newaxis, np.newaxis])
        
        Bt = np.transpose(B, axes=(0, 2, 1))
        Bt_D = np.einsum('eij,jk->eik', Bt, D)
        Bt_D_B = np.einsum('eij,ejk->eik', Bt_D, B)
        Ke_full = Bt_D_B * (self.thickness * areas[:, np.newaxis, np.newaxis])
        
        Ke_full_gpu = cp.array(Ke_full, dtype=cp.float64)
        
        # Extract element diagonal for Jacobi Preconditioner (Shape: E, 6)
        self.K0_diag_gpu = cp.diagonal(Ke_full_gpu, axis1=1, axis2=2)
        
        if self.use_symmetric:
            # Step 4: Upper triangle packing (21 entries per element)
            triu_r, triu_c = np.triu_indices(6)
            Ke0_sym = Ke_full[:, triu_r, triu_c] # (num_elements, 21)
            self.Ke0_gpu_soa = cp.ascontiguousarray(cp.array(Ke0_sym.T, dtype=cp.float64))
        else:
            # Step 1: Full 6x6 layout (36 entries per element)
            Ke0_flat = Ke_full_gpu.reshape(self.num_elements, 36)
            self.Ke0_gpu_soa = cp.ascontiguousarray(Ke0_flat.T)
            
        print(f"      -> 2D Tri3 Tensor Compiled in VRAM: {'21 SoA (Symmetric)' if self.use_symmetric else '36 SoA (Full)'}")

    def _build_jacobi_preconditioner(self, densities_gpu):
        """Assembles inverse diagonal matrix (M^-1) to stabilize PCG."""
        penalty = densities_gpu ** self.p_penalty
        local_diags = self.K0_diag_gpu * penalty[:, cp.newaxis]
        
        M_global = cp.zeros(self.num_dofs, dtype=cp.float64)
        cupyx.scatter_add(M_global, self.elem_dofs_gpu.flatten(), local_diags.flatten())
        
        M_free = M_global[self.free_dofs]
        diag_max = float(cp.max(M_free))
        return 1.0 / cp.maximum(M_free, diag_max * 1e-6)

    def _cpp_spmv_wrapper(self, p_free, densities_gpu):
        """Zero-Copy Bridge to the CUDA C++ SpMV Kernel."""
        self.p_global_buf.fill(0.0)
        self.p_global_buf[self.free_dofs] = p_free
        self.q_global_buf.fill(0.0)
        
        if self.use_symmetric:
            pravaha_cu_engine.launch_spmv_2d_subwarp_sym(
                self.Ke0_gpu_soa.data.ptr,
                densities_gpu.data.ptr,
                self.elem_dofs_gpu_soa.data.ptr,
                self.p_global_buf.data.ptr,
                self.q_global_buf.data.ptr,
                self.num_elements,
                self.num_dofs,
                float(self.p_penalty)
            )
        else:
            pravaha_cu_engine.launch_spmv_2d(
                self.Ke0_gpu_soa.data.ptr,
                densities_gpu.data.ptr,
                self.elem_dofs_gpu_soa.data.ptr,
                self.p_global_buf.data.ptr,
                self.q_global_buf.data.ptr,
                self.num_elements,
                self.num_dofs,
                float(self.p_penalty)
            )
            
        return self.q_global_buf[self.free_dofs]

    def assemble_and_solve(self, densities, F_ext_gpu, is_BOL=True, progress=1.0):
        """Executes Matrix-Free Preconditioned Conjugate Gradient (PCG) solve."""
        solve_start = time.time()
        densities_gpu = cp.asarray(densities, dtype=cp.float64)
        
        F_free = F_ext_gpu[self.free_dofs]
        U_global = cp.zeros(self.num_dofs, dtype=cp.float64)
        
        M_inv = self._build_jacobi_preconditioner(densities_gpu)
        
        A_op = cplinalg.LinearOperator(
            shape=(len(self.free_dofs), len(self.free_dofs)),
            matvec=lambda p: self._cpp_spmv_wrapper(p, densities_gpu),
            dtype=cp.float64
        )
        
        # 3rd-Degree Neumann Polynomial Preconditioner
        def poly_precond(r):
            omega = 0.8
            z = r * M_inv * omega
            v = self._cpp_spmv_wrapper(z, densities_gpu)
            z = z + omega * M_inv * (r - v)
            v = self._cpp_spmv_wrapper(z, densities_gpu)
            z = z + omega * M_inv * (r - v)
            return z

        M_op = cplinalg.LinearOperator(
            shape=(len(self.free_dofs), len(self.free_dofs)),
            matvec=poly_precond,
            dtype=cp.float64
        )
        
        class CGTracker:
            def __init__(self): self.iters = 0
            def __call__(self, xk): self.iters += 1
        tracker = CGTracker()
        
        x0_guess = None
        if is_BOL and self.U_prev_BOL is not None:
            x0_guess = self.U_prev_BOL[self.free_dofs]
        elif not is_BOL and self.U_prev_EOL is not None:
            x0_guess = self.U_prev_EOL[self.free_dofs]
            
        dynamic_tol = self.cg_tol + (0.01 - self.cg_tol) * (1.0 - progress)
        
        U_free, info = cplinalg.cg(A_op, F_free, M=M_op, x0=x0_guess, 
                                   tol=dynamic_tol, maxiter=self.cg_maxiter, callback=tracker)
        
        if tracker.iters == 0 and info > 0:
            tracker.iters = info
            
        U_global[self.free_dofs] = U_free
        
        if is_BOL:
            self.U_prev_BOL = U_global.copy()
        else:
            self.U_prev_EOL = U_global.copy()
            
        solve_time = time.time() - solve_start
        print(f"      -> 2D Matrix-Free PCG ({'BOL' if is_BOL else 'EOL'}): {solve_time:.3f}s ({tracker.iters} iters, tol={dynamic_tol:.1e})")
        return U_global, tracker.iters

    def calculate_sensitivities(self, U_global, densities):
        """Adjoint Method: Evaluates compliance sensitivity via SoA contraction."""
        densities_gpu = cp.asarray(densities, dtype=cp.float64)
        
        # Gather element displacements as (6, num_elements)
        U_elem_soa = U_global[self.elem_dofs_gpu_soa]
        strain_energy = cp.zeros(self.num_elements, dtype=cp.float64)
        
        if self.use_symmetric:
            lut = np.array([
                 0,  1,  2,  3,  4,  5,
                 1,  6,  7,  8,  9, 10,
                 2,  7, 11, 12, 13, 14,
                 3,  8, 12, 15, 16, 17,
                 4,  9, 13, 16, 18, 19,
                 5, 10, 14, 17, 19, 20
            ], dtype=np.int32)
            
            for i in range(6):
                row_dot_u = cp.zeros(self.num_elements, dtype=cp.float64)
                for j in range(6):
                    sym_idx = int(lut[i * 6 + j])
                    row_dot_u += self.Ke0_gpu_soa[sym_idx, :] * U_elem_soa[j, :]
                strain_energy += U_elem_soa[i, :] * row_dot_u
        else:
            for i in range(6):
                row_dot_u = cp.sum(self.Ke0_gpu_soa[i * 6 : i * 6 + 6, :] * U_elem_soa, axis=0)
                strain_energy += U_elem_soa[i, :] * row_dot_u
                
        sens_gpu = -self.p_penalty * (densities_gpu ** (self.p_penalty - 1.0)) * strain_energy
        return cp.asnumpy(sens_gpu)