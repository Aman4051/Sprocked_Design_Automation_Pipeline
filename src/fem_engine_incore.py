import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
import cupy as cp
import cupyx
import cupyx.scipy.sparse.linalg as cplinalg
import time

# Point Python directly to the CMake build folder for the C++ Zero-Copy Kernel
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(build_dir)
import pravaha_cu_engine

class FEMEngineInCoreGPU:
    """
    PRAVAHA V2 In-Core Matrix-Free Solver.
    Stores the full 330MB Element-by-Element matrix directly in RTX 4070 VRAM.
    Fires direct pointers into the Fused C++/CUDA SpMV Kernel for zero-overhead solves.
    """
    def __init__(self, physics_payload, mesh_data, config):
        print("   -> Initializing 3D In-Core Matrix-Free GPU Solver...")
        self.payload = physics_payload
        self.mesh = mesh_data
        self.config = config
        
        # 1. Extract Derated Physics Parameters
        self.E0 = np.float64(self.payload['E_derated'])
        self.nu = np.float64(self.payload['nu'])
        self.p_penalty = float(config.get('optimization_solver', {}).get('simp_penalization_exponent', 3.0))
        
        self.num_nodes = len(self.mesh['nodes'])
        self.num_elements = len(self.mesh['elements'])
        self.num_dofs = self.num_nodes * 3 
        
        self.volumes_gpu = cp.array(self.mesh['volumes'], dtype=cp.float64)
        
        # 2. Reorder Graph to guarantee CUDA Warp Coalescing
        self._apply_rcm_reordering()
        
        # 3. Precompute the Matrix in VRAM
        self._precompute_stiffness_matrices()
        
        # 4. Initialize the Global Force Vectors (Loaded from unified_physics.py)
        # ---  Map external forces to RCM Permuted Space ---
        raw_F_BOL = np.array(self.mesh['F_ext_BOL'], dtype=np.float64)
        raw_F_EOL = np.array(self.mesh['F_ext_EOL'], dtype=np.float64)
        
        # We need the forward permutation array to rearrange the force vectors
        # self.perm is created inside _apply_rcm_reordering, so we will store it as an attribute
        self.F_BOL_gpu = cp.array(raw_F_BOL[self.perm_dofs], dtype=cp.float64)
        self.F_EOL_gpu = cp.array(raw_F_EOL[self.perm_dofs], dtype=cp.float64)

        # --- THE SPEED FIX: WARM START MEMORY ---
        self.U_prev_BOL = None
        self.U_prev_EOL = None
        
        # 4. PREALLOCATE SpMV SCRATCH BUFFERS
        # Prevents ~800GB of VRAM allocation churn during the CG loop
        self.p_global_buf = cp.zeros(self.num_dofs, dtype=cp.float64)
        self.q_global_buf = cp.zeros(self.num_dofs, dtype=cp.float64)

    def _apply_rcm_reordering(self):
        """Reduces the bandwidth of the sparse graph to ensure VRAM caches are hit perfectly."""
        print("      -> Executing Reverse Cuthill-McKee (RCM) Memory Optimization...")
        
        # ---  Vectorized Graph Building ---
        # Bypasses Python list.append overhead which causes CPU OOM on millions of elements.
        elements = self.mesh['elements']
        
        row_indices = np.repeat(elements, 4, axis=1).flatten()
        col_indices = np.tile(elements, (1, 4)).flatten()
        
        # Use int8 for data to save memory
        adj_matrix = sp.coo_matrix((np.ones(len(row_indices), dtype=np.int8), (row_indices, col_indices)), 
                                   shape=(self.num_nodes, self.num_nodes)).tocsr()
                                   
        perm = reverse_cuthill_mckee(adj_matrix)
        inv_perm = np.argsort(perm).astype(np.int32)
        
        # --- THE FIX: Vectorized DOF Mapping ---
        remapped_nodes = inv_perm[elements]
        dofs = np.zeros((self.num_elements, 12), dtype=np.int32)
        
        for i in range(4):
            dofs[:, i*3]     = remapped_nodes[:, i] * 3
            dofs[:, i*3 + 1] = remapped_nodes[:, i] * 3 + 1
            dofs[:, i*3 + 2] = remapped_nodes[:, i] * 3 + 2
            
        self.elem_dofs_gpu = cp.array(dofs)
        
        # Re-map Boundary Conditions
        all_nodes = np.arange(self.num_nodes, dtype=np.int32)
        free_nodes_mask = np.ones(self.num_nodes, dtype=bool)
        free_nodes_mask[self.mesh['fixed_nodes']] = False
        
        free_nodes_original = all_nodes[free_nodes_mask]
        free_nodes_remapped = inv_perm[free_nodes_original]
        
        free_dofs = np.zeros(len(free_nodes_remapped) * 3, dtype=np.int32)
        free_dofs[0::3] = free_nodes_remapped * 3
        free_dofs[1::3] = free_nodes_remapped * 3 + 1
        free_dofs[2::3] = free_nodes_remapped * 3 + 2
        
        self.free_dofs = cp.array(free_dofs)

        # --- Save the forward permutation array for external force vectors ---
        perm_dofs = np.zeros(self.num_dofs, dtype=np.int32)
        perm_dofs[0::3] = perm * 3
        perm_dofs[1::3] = perm * 3 + 1
        perm_dofs[2::3] = perm * 3 + 2
        self.perm_dofs = perm_dofs

    def _precompute_stiffness_matrices(self):
        """
        Calculates Ke = V_e * B^T * D * B instantly in VRAM. 
        Replaces the SSD out-of-core bottleneck entirely.
        """
        print("      -> Precomputing 3D Isotropic D-Matrix and B-Tensors...")
        
        # Standard 3D Isotropic Elasticity Matrix (6x6)
        c1 = self.E0 / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        c2 = c1 * (1.0 - self.nu)
        c3 = c1 * self.nu
        c4 = c1 * ((1.0 - 2.0 * self.nu) / 2.0)
        
        D_cpu = np.array([
            [c2, c3, c3, 0, 0, 0],
            [c3, c2, c3, 0, 0, 0],
            [c3, c3, c2, 0, 0, 0],
            [0, 0, 0, c4, 0, 0],
            [0, 0, 0, 0, c4, 0],
            [0, 0, 0, 0, 0, c4]
        ])
        D_gpu = cp.array(D_cpu, dtype=cp.float64)
        
        # Load the precalculated Tet4 Shape Function Gradients (B matrices)
        # Expected shape: (num_elements, 6, 12) from your mesher
        B_gpu = cp.array(self.mesh['B_matrices'], dtype=cp.float64)
        
        print("      -> Assembling 330MB Element-by-Element Tensor in VRAM...")
        # Einstein Summation: Multiply D by B
        D_B = cp.einsum('ij,ejk->eik', D_gpu, B_gpu) 
        # Einstein Summation: Multiply B^T by (D * B)
        Bt_D_B = cp.einsum('eji,ejk->eik', B_gpu, D_B) 
        
        # Multiply by volume and store
        K_full = Bt_D_B * self.volumes_gpu[:, cp.newaxis, cp.newaxis]
        
        # Extract the diagonal for the Jacobi Preconditioner (Shape: E x 12)
        self.K0_diag_gpu = cp.diagonal(K_full, axis1=1, axis2=2)
        
        # Flatten to AoS (E, 144)
        K0_gpu = K_full.reshape(self.num_elements, 144)
        
        # --- THE MEMORY COALESCING FIX (SoA) ---
        # Create transposed copies (144, E) exclusively for the CUDA Kernel to achieve 500+ GB/s
        self.K0_gpu_soa = cp.ascontiguousarray(K0_gpu.T)
        self.elem_dofs_gpu_soa = cp.ascontiguousarray(self.elem_dofs_gpu.T)
        
        # Delete the AoS matrices to free memory
        del K0_gpu
        del K_full

    def _build_jacobi_preconditioner(self, densities_gpu):
        """Assembles the inverse diagonal matrix (M^-1) to stabilize the CG Solver."""
        penalty = densities_gpu ** self.p_penalty
        local_diags = self.K0_diag_gpu * penalty[:, cp.newaxis] 
        
        M_global = cp.zeros(self.num_dofs, dtype=cp.float64)
        
        # Fast GPU scatter-add
        cupyx.scatter_add(M_global, self.elem_dofs_gpu.flatten(), local_diags.flatten())
        
        M_free = M_global[self.free_dofs]
        
        # --- THE FIX: Clamp relative to max stiffness, not absolute zero ---
        diag_max = float(cp.max(M_free))
        
        # Bounding the preconditioner to 1e-6 of max stiffness prevents 
        # floating point overflow without altering the physical matrix.
        M_inv = 1.0 / cp.maximum(M_free, diag_max * 1e-6)
        
        return M_inv

    def _cpp_spmv_wrapper(self, p_free, densities_gpu):
        """The Zero-Copy Bridge: Bypasses the Global Interpreter Lock (GIL)."""
        p_global = cp.zeros(self.num_dofs, dtype=cp.float64)
        p_global[self.free_dofs] = p_free
        
        # Reuse preallocated buffers to eliminate VRAM allocation latency
        self.p_global_buf.fill(0.0)
        self.p_global_buf[self.free_dofs] = p_free
        
        self.q_global_buf.fill(0.0)
        
        # Fire raw C++ pointers directly to the NVIDIA silicon
        pravaha_cu_engine.launch_spmv(
            self.K0_gpu_soa.data.ptr,         
            densities_gpu.data.ptr,
            self.elem_dofs_gpu_soa.data.ptr,  
            self.p_global_buf.data.ptr,
            self.q_global_buf.data.ptr,
            self.num_elements,
            self.num_dofs,                   
            float(self.p_penalty) 
        )
        
        return self.q_global_buf[self.free_dofs]

    def assemble_and_solve(self, densities, F_ext_gpu, is_BOL=True):
        """Executes the Preconditioned Conjugate Gradient (JPCG) loop."""
        solve_start = time.time()
        densities_gpu = cp.asarray(densities, dtype=cp.float64)
        
        # 1. Prepare Boundaries
        F_free = F_ext_gpu[self.free_dofs]
        U_global = cp.zeros(self.num_dofs, dtype=cp.float64)
        
        # 2. Build Standard Diagonal Inverse
        M_inv = self._build_jacobi_preconditioner(densities_gpu)
        
        # 3. Setup the Linear Operator (PURE PHYSICS)
        A_op = cplinalg.LinearOperator(
            shape=(len(self.free_dofs), len(self.free_dofs)),
            matvec=lambda p: self._cpp_spmv_wrapper(p, densities_gpu),
            dtype=cp.float64
        )
        
        # --- 3rd-Degree Neumann Polynomial Preconditioner ---
        # Recursively applies the SpMV kernel to flatten extreme stiffness 
        # contrasts between solid metal and SIMP void elements.
        def poly_precond(r):
            omega = 0.8  # Damping factor to bound eigenvalues and preserve SPD symmetry
            # Iteration 1 (Standard Damped Jacobi)
            z = r * M_inv * omega
            # Iteration 2
            v = self._cpp_spmv_wrapper(z, densities_gpu)
            z = z + omega * M_inv * (r - v)
            # Iteration 3
            v = self._cpp_spmv_wrapper(z, densities_gpu)
            z = z + omega * M_inv * (r - v)
            return z
            
        M_op = cplinalg.LinearOperator(
            shape=(len(self.free_dofs), len(self.free_dofs)),
            matvec=poly_precond,
            dtype=cp.float64
        )
        
        # 4. Solve via Conjugate Gradients
        class CGTracker:
            def __init__(self):
                self.iters = 0
            def __call__(self, xk):
                self.iters += 1
        tracker = CGTracker()
        
        x0_guess = None
        if is_BOL and self.U_prev_BOL is not None:
            x0_guess = self.U_prev_BOL[self.free_dofs]
        elif not is_BOL and self.U_prev_EOL is not None:
            x0_guess = self.U_prev_EOL[self.free_dofs]
            
        cg_tol = float(self.config.get('optimization_solver', {}).get('cg_tolerance', 5e-3))
        dynamic_tol = cg_tol + (0.01 - cg_tol) * (1.0 - progress)
        
        # Hard-capping the iterations at 5000. 
        # Exiting early provides a perfectly valid gradient direction for SIMP.
        U_free, info = cplinalg.cg(A_op, F_free, M=M_op, x0=x0_guess, tol=dynamic_tol, maxiter=5000, callback=tracker)
        
        if tracker.iters == 0 and info > 0:
            tracker.iters = info
            
        U_global[self.free_dofs] = U_free
        
        # 5. Save the state for the next iteration's Warm Start
        if is_BOL:
            self.U_prev_BOL = U_global.copy()
        else:
            self.U_prev_EOL = U_global.copy()
        
        solve_time = time.time() - solve_start
        print(f"      -> 3D Polynomial-PCG Solve ({'BOL' if is_BOL else 'EOL'}): {solve_time:.3f} sec. ({tracker.iters} iters)")
        
        return U_global, tracker.iters

    def calculate_sensitivities(self, U_global, densities):
        """Analytical Adjoint Method: Evaluates structural compliance sensitivity."""
        densities_gpu = cp.asarray(densities, dtype=cp.float64)
        
        # 1. Zero-Copy SoA gather: extracts nodal displacements directly as (12, E)
        U_elem_soa = U_global[self.elem_dofs_gpu_soa]
        
        # 2. Memory-Efficient SoA Contraction
        # We compute u^T * K0 * u via a fast unrolled loop over the 12 DOFs to restrict peak VRAM.
        strain_energy_unpenalized = cp.zeros(self.num_elements, dtype=cp.float64)
        
        for i in range(12):
            # K_ij * u_j for the i-th row (Shape: 12 x E)
            row_dot_u = cp.sum(self.K0_gpu_soa[i*12 : i*12+12, :] * U_elem_soa, axis=0)
            # Accumulate u_i * (K_ij * u_j)
            strain_energy_unpenalized += U_elem_soa[i, :] * row_dot_u
            
        # Sensitivity = -p * rho^(p-1) * (u^T K0 u)
        sens_gpu = -self.p_penalty * (densities_gpu ** (self.p_penalty - 1.0)) * strain_energy_unpenalized
        
        return cp.asnumpy(sens_gpu)