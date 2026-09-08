import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cupsp
import cupyx.scipy.sparse.linalg as cplinalg

class FEMSolverGPU:
    """
    PRAVAHA V2 Hot-Path 2D Continuum Computational Engine.
    Executes Plane-Stress Tri3 finite element analysis entirely in VRAM.
    """
    def __init__(self, physics_payload, mesh_data, config):
        print("   -> Initializing 2D GPU Plane-Stress FEA Engine...")
        self.payload = physics_payload
        self.mesh = mesh_data
        self.config = config
        
        self.E0 = np.float64(self.payload['E_derated'])
        self.nu = np.float64(self.payload['nu'])
        self.thickness = float(self.config.get('geometry', {}).get('web_thickness_m', 0.007))
        self.p_penalty = float(config.get('optimization_solver', {}).get('simp_penalization_exponent', 3.0))
        
        self.num_nodes = len(self.mesh['nodes'])
        self.num_elements = len(self.mesh['elements'])
        self.num_dofs = self.num_nodes * 2 
        
        self._map_degrees_of_freedom()
        self._precompute_tri3_kinematics()

    def _map_degrees_of_freedom(self):
        """Maps global DOF indices and extracts free DOFs for the sparse solver."""
        fixed_dofs = []
        for n in self.mesh['fixed_nodes']:
            fixed_dofs.extend([n*2, n*2+1])
            
        all_dofs = np.arange(self.num_dofs)
        self.free_dofs = cp.array(np.setdiff1d(all_dofs, fixed_dofs), dtype=cp.int32)
        
        edof = np.zeros((self.num_elements, 6), dtype=np.int32)
        for i in range(3):
            edof[:, 2*i] = self.mesh['elements'][:, i] * 2
            edof[:, 2*i+1] = self.mesh['elements'][:, i] * 2 + 1
            
        self.edof_gpu = cp.array(edof)
        self.iK_gpu = cp.array(np.kron(edof, np.ones((6,1))).flatten())
        self.jK_gpu = cp.array(np.kron(edof, np.ones((1,6))).flatten())

    def _precompute_tri3_kinematics(self):
        """Vectorized 2D Plane-Stress Tensor Assembly."""
        c = self.E0 / (1.0 - self.nu**2)
        D = c * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0]
        ])
        
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
        
        # --- Prevent NaN Division from Degenerate Slivers ---
        A = np.maximum(self.mesh['areas'], 1e-12)
        
        B = np.zeros((self.num_elements, 3, 6))
        B[:, 0, 0] = b1; B[:, 0, 2] = b2; B[:, 0, 4] = b3
        B[:, 1, 1] = c1; B[:, 1, 3] = c2; B[:, 1, 5] = c3
        B[:, 2, 0] = c1; B[:, 2, 1] = b1; B[:, 2, 2] = c2; B[:, 2, 3] = b2; B[:, 2, 4] = c3; B[:, 2, 5] = b3
        B /= (2.0 * A[:, np.newaxis, np.newaxis])
        
        Bt = np.transpose(B, axes=(0, 2, 1))
        Bt_D = np.einsum('eij,jk->eik', Bt, D)
        Bt_D_B = np.einsum('eij,ejk->eik', Bt_D, B)
        Ke0 = Bt_D_B * (self.thickness * A[:, np.newaxis, np.newaxis])
        
        self.Ke0_gpu = cp.array(Ke0, dtype=cp.float64)
        print("      -> 2D Tri3 Tensors Compiled and Loaded to VRAM.")

    def assemble_and_solve(self, densities, F_ext_gpu, is_BOL=True):
        """Executes the Preconditioned Conjugate Gradient (JPCG) loop in 2D."""
        densities_gpu = cp.array(densities, dtype=cp.float64)
        
        # --- DYNAMIC PENALTY ---
        # Fetch the actively ramping p_penalty from the instance, rather than a static init value
        penalty = densities_gpu ** self.p_penalty
        Ke_penalized = self.Ke0_gpu * penalty[:, cp.newaxis, cp.newaxis]
        
        K_global = cupsp.coo_matrix(
            (Ke_penalized.flatten(), (self.iK_gpu, self.jK_gpu)),
            shape=(self.num_dofs, self.num_dofs)
        ).tocsr()
        
        K_free = K_global[self.free_dofs, :][:, self.free_dofs]
        F_free = F_ext_gpu[self.free_dofs]
        
        diag_K = K_free.diagonal()
        M_inv = 1.0 / cp.maximum(diag_K, 1e-12)
        
        # --- 3rd-Degree Neumann Polynomial Preconditioner (2D) ---
        def poly_precond_2d(r):
            omega = 0.8 
            z = r * M_inv * omega
            
            Kz = K_free.dot(z)
            z = z + omega * M_inv * (r - Kz)
            
            Kz = K_free.dot(z)
            z = z + omega * M_inv * (r - Kz)
            return z
            
        M_op = cplinalg.LinearOperator(
            shape=(len(self.free_dofs), len(self.free_dofs)),
            matvec=poly_precond_2d,
            dtype=cp.float64
        )
        
        U_free, info = cplinalg.cg(K_free, F_free, M=M_op, tol=1e-3, maxiter=2500)
        
        if info > 0:
            print(f"      [⚠] 2D CG Solver halted at {info} iterations (Sliver Matrix Singularity).")
            pass
            
        U_global = cp.zeros(self.num_dofs, dtype=cp.float64)
        U_global[self.free_dofs] = U_free
        
        return U_global

    def calculate_sensitivities(self, U_global, densities):
        """Analytical Adjoint Method: Evaluates structural compliance sensitivity."""
        densities_gpu = cp.array(densities, dtype=cp.float64)
        
        U_elem = U_global[self.edof_gpu]
        Ke0_u = cp.einsum('eij,ej->ei', self.Ke0_gpu, U_elem)
        ue_Ke0_ue = cp.einsum('ei,ei->e', U_elem, Ke0_u)
        
        sens_gpu = -self.p_penalty * (densities_gpu ** (self.p_penalty - 1.0)) * ue_Ke0_ue
        
        return cp.asnumpy(sens_gpu)