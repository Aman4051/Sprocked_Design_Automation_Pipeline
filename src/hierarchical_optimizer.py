import time
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupyx.scipy.sparse as cupsp
from scipy.spatial import cKDTree

class HierarchicalTopologyOptimizer:
    """
    PRAVAHA V2 Hierarchical Memetic Optimizer.
    Phase 1: Blindingly fast 2D planar OC optimization (The Blueprint).
    Phase 2: Z-Axis KD-Tree mathematical extrusion.
    Phase 3: High-fidelity 3D Tet4 spatial polish via C++ SpMV.
    """
    def __init__(self, config, fem_2d, fem_3d):
        print("   -> Initializing Hierarchical Memetic Optimizer...")
        self.config = config
        self.fem_2d = fem_2d
        self.fem_3d = fem_3d
        
        self.target_vol_fraction = float(config.get('constraints', {}).get('target_volume_fraction', 0.40))
        self.max_iter_2d = int(config.get('optimization_solver', {}).get('max_iterations_2d', 50))
        self.max_iter_3d = int(config.get('optimization_solver', {}).get('max_iterations_3d', 25))
        
        self.penalty_ramp = float(config.get('optimization_solver', {}).get('penalty_ramp_iterations', 50.0))
        
        self.telemetry = []

    def _execute_oc_loop(self, fem_engine, rho_init, max_iter, is_3d=False):
        """
        The Unified Optimality Criteria (OC) mathematical engine.
        Integrates Heaviside Projection (Pillar 3), Cyclic Mapping (Pillar 2),
        and the new Rotational Inertia Minimization Objective.
        """
        phase_name = "3D SPATIAL POLISH" if is_3d else "2D PLANAR SPRINT"
        print(f"\n{'='*60}")
        print(f" 🚀 INITIATING {phase_name} (OC SOLVER)")
        print(f"{'='*60}")
        
        vols = fem_engine.mesh.get('volumes', fem_engine.mesh.get('areas'))
        vols_gpu = cp.array(vols, dtype=cp.float64)
        
        b_min_gpu = cp.array(fem_engine.mesh['bounds_min'], dtype=cp.float64)
        b_max_gpu = cp.array(fem_engine.mesh['bounds_max'], dtype=cp.float64)
        
        V_solid = cp.sum(vols_gpu[b_min_gpu == 1.0])
        V_free  = cp.sum(vols_gpu[(b_min_gpu < 1.0) & (b_max_gpu > 0.01)])
        target_vol_abs = V_solid + (self.target_vol_fraction * V_free)
        
        rho = cp.array(rho_init, dtype=cp.float64)
        
        # The 3D engine stores RCM-permuted forces. The 2D engine uses raw node order.
        if hasattr(fem_engine, 'F_BOL_gpu'):
            F_BOL_gpu = fem_engine.F_BOL_gpu
            F_EOL_gpu = fem_engine.F_EOL_gpu
        else:
            F_BOL_gpu = cp.array(fem_engine.mesh['F_ext_BOL'], dtype=cp.float64)
            F_EOL_gpu = cp.array(fem_engine.mesh['F_ext_EOL'], dtype=cp.float64)

        if not hasattr(fem_engine, 'H_gpu'):
            print("      -> Initializing Persistent Static Filter & Symmetry Maps...")
            fem_engine.H_gpu, fem_engine.H_sum_gpu = self._build_filter_matrix(fem_engine, is_3d)
        else:
            print("      -> Persistent Filter Matrix detected. Skipping rebuild.")
            
        if not hasattr(fem_engine, 'cyclic_map_gpu'):
            self._build_cyclic_mapping(fem_engine)
            
        H_gpu = fem_engine.H_gpu
        H_sum_gpu = fem_engine.H_sum_gpu
        
        nodes = fem_engine.mesh['nodes']
        elements = fem_engine.mesh['elements']
        centroids = np.mean(nodes[elements], axis=1)
        r_cents = np.hypot(centroids[:, 0], centroids[:, 1])
        r_cents_gpu = cp.array(r_cents, dtype=cp.float64)
        R_max = float(cp.max(r_cents_gpu))
        
        # Compute the baseline sensitivity of Inertia (positive values: V * (r/Rmax)^2 )
        S_I_gpu = vols_gpu * (r_cents_gpu / R_max)**2
        max_I = float(cp.max(S_I_gpu))
        
        # Load the user's inertia penalty from YAML, default to 0.15 for aggressive rim-weight reduction
        gamma = float(self.config.get('optimization_solver', {}).get('inertia_weight', 0.15))
        if gamma > 0:
            print(f"      -> ⚙️ Inertia Minimization Active. Rim Mass Penalty: {gamma*100:.1f}%")
        
        cp.get_default_pinned_memory_pool().free_all_blocks()

        def _project(field, eta, beta_val):
            num_ = cp.tanh(beta_val * eta) + cp.tanh(beta_val * (field - eta))
            den_ = cp.tanh(beta_val * eta) + cp.tanh(beta_val * (1.0 - eta))
            return num_ / den_, den_

        for iteration in range(max_iter):
            loop_start = time.time()
            
            progress = min(1.0, iteration / self.penalty_ramp)
            dynamic_move_limit = max(0.02, 0.20 - (0.18 * progress))
            dynamic_eta_damping = max(0.20, 0.50 - (0.30 * progress))
            
            current_penalty = min(3.0, 1.0 + (2.0 * iteration / self.penalty_ramp))
            fem_engine.p_penalty = current_penalty
            
            beta = min(16.0, 1.0 * (iteration // (self.penalty_ramp / 3.0) + 1))

            rho_bar = H_gpu.dot(rho) / H_sum_gpu

            rho_blue, _den_blue = _project(rho_bar, 0.5, beta)                
            rho_erode, den_erode = _project(rho_bar, fem_engine.eta_erode, beta)    

            rho_blue = cp.maximum(rho_blue, 0.01)
            rho_erode = cp.maximum(rho_erode, 0.01)

            # --- SOLVE PHASE VRAM TRACKING ---
            U_BOL = fem_engine.assemble_and_solve(rho_erode, F_BOL_gpu, is_BOL=True)
            U_EOL = fem_engine.assemble_and_solve(rho_erode, F_EOL_gpu, is_BOL=False)
            vram_solve = cp.get_default_memory_pool().used_bytes() / (1024**3)
            
            comp_BOL = float(cp.sum(F_BOL_gpu[fem_engine.free_dofs] * U_BOL[fem_engine.free_dofs]))
            comp_EOL = float(cp.sum(F_EOL_gpu[fem_engine.free_dofs] * U_EOL[fem_engine.free_dofs]))
            total_comp = comp_BOL + comp_EOL
            
            # --- SENSITIVITY PHASE VRAM TRACKING ---
            sens_BOL = fem_engine.calculate_sensitivities(U_BOL, rho_erode)
            sens_EOL = fem_engine.calculate_sensitivities(U_EOL, rho_erode)
            total_sens_raw = cp.array(sens_BOL + sens_EOL, dtype=cp.float64)
            
            d_erode = (beta * (1.0 - cp.tanh(beta * (rho_bar - fem_engine.eta_erode))**2)) / den_erode
            total_sens_raw *= d_erode
            
            import cupyx
            summed_sens = cp.zeros_like(total_sens_raw)
            cupyx.scatter_add(summed_sens, fem_engine.cyclic_map_gpu, total_sens_raw)
            symmetric_sens_raw = summed_sens[fem_engine.cyclic_map_gpu] / fem_engine.num_sectors
            
            design_mask_gpu = (b_min_gpu < 1.0)
            total_sens_raw = cp.where(design_mask_gpu, symmetric_sens_raw, total_sens_raw)
            
            filtered_sens = H_gpu.dot(total_sens_raw) / H_sum_gpu
            vram_sens = cp.get_default_memory_pool().used_bytes() / (1024**3)
            
            # Dual-Point Tracking: Capture the true peak across solve and sensitivity phases
            current_vram = max(vram_solve, vram_sens)
            
            if gamma > 0.0:
                max_C = float(cp.max(cp.abs(filtered_sens)))
                # Dynamically scale the inertia gradient so it competes with the structural strain energy
                scaled_S_I = S_I_gpu * (max_C / max_I) * gamma
                # filtered_sens is negative. Adding scaled_S_I pushes rim voxels toward positive (deletion).
                filtered_sens += scaled_S_I
            
            l1 = 0.0
            l2 = float(cp.max(cp.abs(filtered_sens) / vols_gpu)) * 2.0 
            l2 = max(l2, 1e9) 
            
            rho_new = cp.zeros_like(rho)
            
            while (l2 - l1) > 1e-5:
                l_mid = 0.5 * (l1 + l2)
                
                B = (-filtered_sens) / (l_mid * vols_gpu)
                B = cp.maximum(B, 1e-10) 
                
                # Apply Dynamic Damping & Move Limits
                B_pow = B ** dynamic_eta_damping
                
                rho_upper = cp.minimum(b_max_gpu, rho + dynamic_move_limit)
                rho_lower = cp.maximum(cp.maximum(b_min_gpu, 0.01), rho - dynamic_move_limit)
                rho_new = cp.maximum(rho_lower, cp.minimum(rho_upper, rho * B_pow))
                
                rho_new = rho_new[fem_engine.cyclic_map_gpu]
                
                if cp.sum(rho_new * vols_gpu) > target_vol_abs:
                    l1 = l_mid
                else:
                    l2 = l_mid
                    
            change = float(cp.max(cp.abs(rho_new - rho)))
            rho = rho_new.copy()
            loop_time = time.time() - loop_start
            
            rho_bar_check = H_gpu.dot(rho) / H_sum_gpu
            rho_blue_check, _ = _project(rho_bar_check, 0.5, beta)
            rho_blue_check = cp.maximum(rho_blue_check, 0.01)
            vol_match = float(cp.sum(rho_blue_check * vols_gpu) / target_vol_abs) * 100.0
            
            print(f"      Iter {iteration+1:03d} | Comp: {total_comp:.2e} | Penalty: {current_penalty:.2f} | Max Δρ: {change:.4f} | Vol Match: {vol_match:.1f}% | Time: {loop_time:.2f}s")
            
            self.telemetry.append((1 if is_3d else 0, iteration + 1, total_comp, change, loop_time, current_vram))
            
            # STRICT CONTROL: Obey the central OC tolerance
            oc_tol = float(self.config.get('optimization_solver', {}).get('oc_tolerance', 0.01))
            
            # THE FIX: Only allow convergence checks AFTER the SIMP penalty has fully hardened to 3.0
            if change < oc_tol and iteration >= self.penalty_ramp:
                print(f"   ✅ {'3D' if is_3d else '2D'} Topology Converged (Δρ < {oc_tol}) in {iteration+1} iterations.")
                break

        beta_final = 16.0
        rho_bar_final = H_gpu.dot(rho) / H_sum_gpu
        rho_blue_final, _ = _project(rho_bar_final, 0.5, beta_final)
        
        rho_erode_final, _ = _project(rho_bar_final, fem_engine.eta_erode, beta_final)
        rho_dilate_final, _ = _project(rho_bar_final, fem_engine.eta_dilate, beta_final)
        rho_blue_final = cp.maximum(rho_blue_final, 0.01)

        void_blue = float(cp.sum(((rho_blue_final < 0.5) & (b_max_gpu > 0.01)).astype(cp.float64)))
        void_dilate = float(cp.sum(((rho_dilate_final < 0.5) & (b_max_gpu > 0.01)).astype(cp.float64)))
        if void_blue > 0 and void_dilate < 0.5 * void_blue:
            print(f"      [⚠] Length-scale audit: dilated void count collapsed from "
                  f"{int(void_blue)} to {int(void_dilate)} elements -- some pockets are "
                  f"likely narrower than the tool clearance implied by eta_dilate="
                  f"{fem_engine.eta_dilate:.3f}. Consider a finer mesh or larger filter radius "
                  f"before trusting this run's manufacturability.")

        return cp.asnumpy(rho_blue_final)
    
    def _project_2d_to_3d_guess(self, rho_2d):
        """
        Phase 2: The Spatial Preconditioner.
        Uses a KD-Tree to mathematically extrude the 2D planar topology into the Z-axis 
        of the 3D voxel grid, bypassing hours of 3D brute-force matrix solving.
        """
        print("\n" + "="*60)
        print(" 🌉 PHASE 2: EXTRUDING 2D TOPOLOGY TO 3D VOXEL PRECONDITIONER")
        print("="*60)
        
        nodes_2d = self.fem_2d.mesh['nodes']
        nodes_3d = self.fem_3d.mesh['nodes']
        
        p1_2d = nodes_2d[self.fem_2d.mesh['elements'][:, 0]]
        p2_2d = nodes_2d[self.fem_2d.mesh['elements'][:, 1]]
        p3_2d = nodes_2d[self.fem_2d.mesh['elements'][:, 2]]
        centroids_2d = (p1_2d + p2_2d + p3_2d) / 3.0
        
        p1_3d = nodes_3d[self.fem_3d.mesh['elements'][:, 0]][:, :2]
        p2_3d = nodes_3d[self.fem_3d.mesh['elements'][:, 1]][:, :2]
        p3_3d = nodes_3d[self.fem_3d.mesh['elements'][:, 2]][:, :2]
        p4_3d = nodes_3d[self.fem_3d.mesh['elements'][:, 3]][:, :2]
        centroids_3d = (p1_3d + p2_3d + p3_3d + p4_3d) / 4.0
        
        print("      -> Building spatial mapping tree...")
        tree_2d = cKDTree(centroids_2d)
        _, indices = tree_2d.query(centroids_3d)
        
        rho_3d_guess = rho_2d[indices]
        
        rho_3d_guess = np.maximum(self.fem_3d.mesh['bounds_min'], 
                                  np.minimum(self.fem_3d.mesh['bounds_max'], rho_3d_guess))
                                  
        print(f"   ✅ Successfully mapped {len(rho_2d)} 2D pixels onto {len(rho_3d_guess)} 3D voxels.")
        return rho_3d_guess

    def execute_pipeline(self):
        """Orchestrates the entire Hierarchical Topology Generation."""
        start_time = time.time()
        
        b_min_2d = self.fem_2d.mesh['bounds_min']
        b_max_2d = self.fem_2d.mesh['bounds_max']
        initial_rho_2d = np.where(b_min_2d == 1.0, 1.0, 
                                  np.where(b_max_2d == 0.01, 0.01, self.target_vol_fraction))
        
        rho_2d_optimized = self._execute_oc_loop(self.fem_2d, initial_rho_2d, self.max_iter_2d, is_3d=False)
        
        rho_3d_guess = self._project_2d_to_3d_guess(rho_2d_optimized)
        
        rho_3d_final = self._execute_oc_loop(self.fem_3d, rho_3d_guess, self.max_iter_3d, is_3d=True)
        
        total_time = time.time() - start_time
        print(f"\n🎉 HIERARCHICAL OPTIMIZATION COMPLETE IN {total_time:.2f} SECONDS.")
        
        return rho_3d_final, self.telemetry
    
    def _calibrate_length_scale_thresholds(self, filter_radius_m, mesh_spacing_m, r_min_void, r_min_solid, beta_final=16.0):
        h = float(np.clip(mesh_spacing_m, filter_radius_m / 40.0, filter_radius_m / 8.0))
        L = 6.0 * filter_radius_m
        x = np.arange(-L, L + h, h)

        rho_true = (x <= 0.0).astype(np.float64)

        d = np.abs(x[:, None] - x[None, :])
        W = np.maximum(0.0, filter_radius_m - d)
        rho_bar = (W @ rho_true) / np.sum(W, axis=1)

        def _proj(field, eta, beta):
            num_ = np.tanh(beta * eta) + np.tanh(beta * (field - eta))
            den_ = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
            return num_ / den_

        etas = np.linspace(0.03, 0.97, 377)
        offsets = np.full_like(etas, np.nan)

        for i, eta in enumerate(etas):
            rho_tilde = _proj(rho_bar, eta, beta_final)
            s = rho_tilde - 0.5
            cross = np.where(np.diff(np.sign(s)) != 0)[0]
            if len(cross) == 0:
                continue
            k = cross[0]
            x0, x1 = x[k], x[k + 1]
            y0, y1 = s[k], s[k + 1]
            x_iso = x0 - y0 * (x1 - x0) / (y1 - y0)
            offsets[i] = -x_iso  

        valid = ~np.isnan(offsets)
        etas_v, offsets_v = etas[valid], offsets[valid]

        erode_mask = etas_v >= 0.5
        e_eta, e_off = etas_v[erode_mask], offsets_v[erode_mask]
        order = np.argsort(e_off)
        eta_erode = float(np.interp(r_min_solid, e_off[order], e_eta[order]))

        dilate_mask = etas_v <= 0.5
        d_eta, d_off = etas_v[dilate_mask], offsets_v[dilate_mask]
        order = np.argsort(d_off)
        eta_dilate = float(np.interp(-r_min_void, d_off[order], d_eta[order]))

        return float(np.clip(eta_dilate, 0.02, 0.48)), float(np.clip(eta_erode, 0.52, 0.98))

    def _build_filter_matrix(self, fem_engine, is_3d=False):
        print("      -> Building Ultra-Low-RAM Direct CSR Filter...")
        nodes = fem_engine.mesh['nodes']
        elements = fem_engine.mesh['elements']
        centroids = np.mean(nodes[elements], axis=1)
        num_elem = len(centroids)

        cnc_radius_m = float(self.config.get('manufacturing_routing', {}).get('cnc_endmill_radius_mm', 2.0)) / 1000.0
        min_web_m = float(self.config.get('manufacturing_routing', {}).get('min_structural_web_width_m', 0.008))

        r_min_void = cnc_radius_m
        r_min_solid = min_web_m / 2.0

        target_r_min = max(r_min_void, r_min_solid) * 2.5

        tree = cKDTree(centroids)
        sample_n = min(2000, num_elem)
        sample_idx = np.random.default_rng(0).choice(num_elem, sample_n, replace=False)
        nn_dist, _ = tree.query(centroids[sample_idx], k=2)
        mesh_spacing_m = float(np.median(nn_dist[:, 1]))

        fem_engine.eta_dilate, fem_engine.eta_erode = self._calibrate_length_scale_thresholds(
            filter_radius_m=target_r_min,
            mesh_spacing_m=mesh_spacing_m,
            r_min_void=r_min_void,
            r_min_solid=r_min_solid,
        )
        print(f"      -> Calibrated length-scale thresholds (measured, mesh_spacing={mesh_spacing_m*1000:.3f}mm): "
              f"eta_dilate={fem_engine.eta_dilate:.3f} (void >= {r_min_void*1000:.2f}mm), "
              f"eta_erode={fem_engine.eta_erode:.3f} (solid >= {r_min_solid*1000:.2f}mm)")
        
        k_neighbors = 256 if is_3d else 128
        
        batch_size = 50000 
        num_batches = int(np.ceil(num_elem / batch_size))
        
        max_nnz = num_elem * k_neighbors
        
        csr_data = np.zeros(max_nnz, dtype=np.float32)
        csr_indices = np.zeros(max_nnz, dtype=np.int32)
        csr_indptr = np.zeros(num_elem + 1, dtype=np.int32)
        
        nnz_ptr = 0
        
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, num_elem)
            
            dist, idx = tree.query(centroids[start:end], k=k_neighbors, workers=-1)
            
            for i in range(end - start):
                global_i = start + i
                d = dist[i]
                valid = d <= target_r_min
                
                n_valid = np.sum(valid)
                
                if n_valid > 0:
                    weights = np.maximum(0.0, target_r_min - d[valid])
                    
                    csr_data[nnz_ptr : nnz_ptr + n_valid] = weights
                    csr_indices[nnz_ptr : nnz_ptr + n_valid] = idx[i][valid]
                    nnz_ptr += n_valid
                    
                csr_indptr[global_i + 1] = nnz_ptr
            
            print(f"         [Batch {b+1}/{num_batches}] Processed {end} elements... (NNZ: {nnz_ptr})")

        csr_data = csr_data[:nnz_ptr]
        csr_indices = csr_indices[:nnz_ptr]
        
        print("      -> Compiling CSR Matrix (Zero-Copy)...")
        H_cpu = sp.csr_matrix((csr_data, csr_indices, csr_indptr), shape=(num_elem, num_elem), copy=False)
        
        H_cpu.sum_duplicates()
        H_cpu.eliminate_zeros()
        H_cpu.sort_indices()
        
        H_sum = np.array(H_cpu.sum(axis=1)).flatten()
        
        data_1d = H_cpu.data
        indices_1d = H_cpu.indices
        indptr_1d = H_cpu.indptr
        
        del H_cpu
        import gc
        gc.collect()
        
        print("      -> Streaming Filter Arrays to GPU VRAM via PCIe Chunks...")
        
        data_gpu = cp.empty(len(data_1d), dtype=cp.float32)
        indices_gpu = cp.empty(len(indices_1d), dtype=cp.int32)
        indptr_gpu = cp.array(indptr_1d, dtype=cp.int32) 
        
        chunk_size = 10000000
        total_chunks = int(np.ceil(len(data_1d) / chunk_size))
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(data_1d))
            
            data_gpu[start_idx:end_idx] = cp.asarray(data_1d[start_idx:end_idx])
            indices_gpu[start_idx:end_idx] = cp.asarray(indices_1d[start_idx:end_idx])
            
        print("      -> Streaming Complete. Reassembling CSR Matrix natively...")
        
        del data_1d, indices_1d, indptr_1d
        gc.collect()
        
        H_gpu = cupsp.csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(num_elem, num_elem))
        
        return H_gpu, cp.array(H_sum)
    
    def _determine_structural_sector_count(self, num_bolts, num_teeth):
        """
        Dynamically calculates the optimal number of structural cyclic sectors (N_s).
        Enforces GCD rules for Lug-Centric Hubs (Type D) or the Hunting Spoke algorithm 
        for Continuous Hubs (Type A, B, C) to prevent acoustic resonance.
        """
        if num_bolts > 0:
            print("      -> Hub Topology Detected: Type D (Lug-Centric/Bolted)")
            best_ns = 1
            for n in range(num_bolts, 2, -1):
                if num_bolts % n == 0:
                    best_ns = n
                    break
            
            N_s = max(best_ns, 3)
            print(f"      -> Enforcing GCD Geometric Constraint: Sector Count N_s = {N_s}")
            return N_s
            
        else:
            print("      -> Hub Topology Detected: Type A/B/C (Continuous)")
            priority_spokes = [5, 7, 3, 9]
            
            for N_s in priority_spokes:
                if num_teeth % N_s != 0:
                    print(f"      -> Hunting Spoke Failsafe Activated: Selected N_s = {N_s} (Breaks Harmonic Symmetry)")
                    return N_s
            
            N_s = 11
            while True:
                if num_teeth % N_s != 0:
                    print(f"      -> Industrial Fallback Activated: Selected N_s = {N_s}")
                    return N_s
                N_s += 2 

    def _build_cyclic_mapping(self, fem_engine):
        """
        Calculates the KD-Tree twin map to enforce instant rotational symmetry.
        Universally handles both 2D Tri3 and 3D Tet4 meshes using the dynamically calculated N_s.
        """
        print("      -> Compiling Pillar 2: Cyclic Symmetry Mapping Matrix...")
        nodes = fem_engine.mesh['nodes']
        elements = fem_engine.mesh['elements']
        
        geom_config = self.config.get('geometry', {})
        sprocket_type = str(geom_config.get('sprocket_type', 'C')).upper()
        
        # --- THE FIX: Ignore YAML bolt counts for Hubless Types ---
        num_bolts = int(geom_config.get('num_bolts', 0)) if sprocket_type == 'D' else 0
        num_teeth = int(geom_config.get('num_teeth', 39))
        
        self.N_sectors = self._determine_structural_sector_count(num_bolts, num_teeth)
        sector_angle = (2.0 * np.pi) / self.N_sectors
        
        # Align AI Symmetry with CAM Phase-Shift
        offset_angle = (sector_angle / 2.0) if num_bolts > 0 else 0.0

        elem_nodes = nodes[elements]
        centroids_x = np.mean(elem_nodes[:, :, 0], axis=1)
        centroids_y = np.mean(elem_nodes[:, :, 1], axis=1)

        radii = np.hypot(centroids_x, centroids_y)
        raw_thetas = np.arctan2(centroids_y, centroids_x)
        
        shifted_thetas = np.mod(raw_thetas - offset_angle, 2.0 * np.pi)

        master_mask = shifted_thetas <= (sector_angle + 1e-4) 
        master_indices = np.where(master_mask)[0]
        
        master_coords = np.column_stack((centroids_x[master_indices], centroids_y[master_indices]))
        tree = cKDTree(master_coords)

        folded_shifted_thetas = np.mod(shifted_thetas, sector_angle)
        mapped_physical_thetas = folded_shifted_thetas + offset_angle
        
        folded_coords = np.column_stack((radii * np.cos(mapped_physical_thetas), radii * np.sin(mapped_physical_thetas)))

        _, nearest_local_idx = tree.query(folded_coords)

        cyclic_map_cpu = master_indices[nearest_local_idx]
        
        fem_engine.cyclic_map_gpu = cp.array(cyclic_map_cpu, dtype=cp.int32)
        fem_engine.num_sectors = self.N_sectors
        
        print(f"      -> Symmetry Enforced (Phase-Shifted): Reduced {len(elements)} elements to {len(master_indices)} master design variables.")