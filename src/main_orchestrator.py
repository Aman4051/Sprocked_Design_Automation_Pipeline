import os
import time
import yaml
import numpy as np
import csv
import json
import subprocess 
from datetime import datetime
from shapely.geometry import Point
from shapely.prepared import prep
from manufacturing_export import SprocketCAMExporter

# --- PRAVAHA V2 ARCHITECTURE IMPORTS ---
from ansi_profile_generator import ProceduralANSIGenerator
from unified_physics import UnifiedPhysicsEngine
from gmsh_continuum_mesher import GmshContinuumMesher3D as DualContinuumMesher
from fem_engine_incore import FEMEngineInCoreGPU
from fem_gpu_solver import FEMSolverGPU as FEMEngineInCoreGPU2D 
from hierarchical_optimizer import HierarchicalTopologyOptimizer
from research_telemetry import AcademicPassportBuilder

class PravahaOrchestratorV2:
    """
    The Central Nervous System of the PRAVAHA V2 Suite.
    Executes the Hierarchical Memetic optimization pipeline.
    Stops at the .npz mathematical array for decoupled DFM post-processing.
    """
    def __init__(self, config_path, materials_path):
        print("\n" + "="*70)
        print(" ⚙️ INITIALIZING PRAVAHA V2 HIERARCHICAL ORCHESTRATOR")
        print("="*70)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(materials_path, 'r') as f:
            self.materials_db = yaml.safe_load(f)
            
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../results/run_{timestamp}"))
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.ml_dir = os.path.join(self.run_dir, "ml_training_data")
        os.makedirs(self.ml_dir, exist_ok=True)
        self.ml_sample_counter = 0
        
        self.npz_path = os.path.join(self.run_dir, "topology_3d_BEST_TRIAL.npz")
        self.csv_path = os.path.join(self.run_dir, "telemetry.csv")
        self.json_path = os.path.join(self.run_dir, "run_metadata.json")
        
        self.run_metadata = {}

    def execute_cold_path(self):
        """Phase 1: Procedural CAD, Unified Physics, and Dual Meshing."""
        print("\n[1/4] EXECUTING COLD-PATH PHYSICS & MESHING...")
        
        ansi_gen = ProceduralANSIGenerator(self.config)
        self.profile_coords, self.gap_coords = ansi_gen.generate_and_export() 
        
        physics = UnifiedPhysicsEngine(self.config, self.materials_db)
        self.payload = physics.build_physics_payload()
        
        mesher = DualContinuumMesher(self.config, self.payload, 
                                     profile_coords=self.profile_coords, 
                                     gap_coords=self.gap_coords)
                                     
        self.mesh_2d = mesher.build_2d_mesh()
        self.mesh_3d = mesher.build_3d_mesh()

    def evaluate_analytical_fos(self, v_mid):
        """The 'Sanity Check' Judge using Macroscopic Fatigue Limits."""
        import math
        print("      -> [Analytical Judge] Calculating Macroscopic Fatigue Limits...")
        
        Sy = self.payload['Sy_derated']
        T_max = self.payload['T_max']
        N_driven = self.config.get('geometry', {}).get('num_teeth', 39)
        N_engaged = self.payload['N_engaged']
        
        geom = self.config.get('geometry', {})
        R_hub = float(geom.get('bolt_pcd_m', 0.081)) / 2.0
        R_pitch = self.payload['R_pitch']
        thickness = float(geom.get('web_thickness_m', 0.007))
        
        A_cross_section = 2.0 * math.pi * R_hub * thickness
        A_actual = A_cross_section * v_mid
        
        Torque_Nm = T_max * R_pitch
        F_hub = Torque_Nm / R_hub
        
        mu = float(self.config.get('tribology', {}).get('friction_coefficient', 0.05))
        alpha = (2.0 * math.pi) / N_driven
        friction_angle = math.atan(mu)
        phi = math.radians(17.0 - (64.0 / N_driven)) 
        
        K = math.sin(phi + alpha + friction_angle) / math.sin(phi - friction_angle)
        N_effective = sum([(1.0 / K)**(i - 1) for i in range(1, N_engaged + 1)])
        
        active_strut_area = A_actual 
        
        tau_transverse = F_hub / active_strut_area
        
        sigma_axial = tau_transverse * 0.5 
        
        vm_analytical = math.sqrt((sigma_axial**2) + 3.0 * (tau_transverse**2))
        FoS_analytical = Sy / vm_analytical
        
        return FoS_analytical
    def evaluate_analytical_tooth_failsafe(self):
        """The User-Defined Lewis Bending Failsafe."""
        import math
        T_max = self.payload['T_max']
        N_total = self.config.get('num_teeth', 39)
        mu = 0.15

        alpha = (2.0 * math.pi) / N_total
        rho_angle = math.atan(mu)
        phi = math.radians(17.0 - (64.0 / N_total))
        K = math.sin(phi + alpha + rho_angle) / math.sin(phi - rho_angle)

        F_n1 = T_max * (math.sin(alpha) / math.sin(phi + alpha + rho_angle))

        b = self.config.get('rim_thickness_m', 0.007)
        t = 0.00635  
        h = 0.00425  
        
        F_bend = F_n1 * math.cos(phi)

        sigma_nom = (6 * F_bend * h) / (b * t**2)
        Kt = 2.0 
        sigma_peak = sigma_nom * Kt

        Sy = self.payload['Sy_derated']
        FoS_analytical_tooth = Sy / sigma_peak

        return FoS_analytical_tooth

    def _calculate_rotational_inertia(self, densities):
        """Calculates the physical Mass Moment of Inertia (Iz) in kg*m^2."""
        mat_name = self.config.get('active_material', 'al_7075_t6')
        rho_mat = float(self.materials_db.get(mat_name, {}).get('density_kg_m3', 2810.0))
        
        nodes = self.mesh_3d['nodes'] 
        elements = self.mesh_3d['elements']
        volumes = self.mesh_3d['volumes']
        
        elem_nodes = nodes[elements]
        centroids = np.mean(elem_nodes, axis=1)
        
        r_sq = centroids[:, 0]**2 + centroids[:, 1]**2
        
        I_z = np.sum(densities * volumes * rho_mat * r_sq)
        return float(I_z)
    
    def execute_outer_loop(self):
        """Phase 2: Incumbent-Driven Sequential Approximate Optimization Outer Loop."""
        print("\n[2/4] WAKING UP NVIDIA RTX GPU SOLVERS...")
        
        self.fem_2d = FEMEngineInCoreGPU2D(self.payload, self.mesh_2d, self.config)
        self.fem_3d = FEMEngineInCoreGPU(self.payload, self.mesh_3d, self.config)
        
        print("\n[3/4] LAUNCHING THE AUTONOMOUS DECOUPLED OUTER LOOP...")
        
        cnstr = self.config.get('constraints', {})
        target_web_fos = cnstr.get('target_web_fos', 2.20)
        target_shear_fos = cnstr.get('target_shear_fos', 1.50)
        max_in_plane = cnstr.get('max_in_plane_deflection_m', 1.0e-4)
        max_axial = cnstr.get('max_axial_deflection_m', 2.5e-4)
        minimum_tooth_fos = cnstr.get('minimum_tooth_fos', 1.10)
        max_iterations = cnstr.get('max_outer_iterations', 10)

        gamma_iters = int(self.config.get('optimization_solver', {}).get('gamma_tuning_iterations', 3))
        
        v_floor = float(cnstr.get('min_volume_fraction', 0.20))
        v_ceiling = 0.85 
        v_mid = (v_floor + v_ceiling) / 2.0
        
        self.optimal_densities = None
        self.telemetry = []
        
        self.best_feasible_state = None
        self.best_feasible_inertia = float('inf') # Track by true physical objective
        self.feasible_vault = [] # Archival ledger of all successful configurations
        
        # --- PLATEAU DETECTION VARIABLES ---
        self.stalled_counter = 0
        self.previous_margin = 0.0
        self.previous_margin_tooth = 0.0
        self.previous_margin_stress = 0.0
        self.previous_margin_in_plane = 0.0
        self.previous_margin_axial = 0.0
        self.previous_v_mid = 0.0
        self.last_direction = "UP"
        max_stalled = int(self.config.get('constraints', {}).get('max_stalled_volume_additions', 3))
        plateau_min_improvement = float(self.config.get('constraints', {}).get('plateau_min_margin_improvement', 0.015))
        plateau_scramble_multiplier = float(self.config.get('constraints', {}).get('plateau_scramble_volume_multiplier', 0.75))
        
        for outer_iter in range(max_iterations + 1):
            if outer_iter == max_iterations:
                if self.best_feasible_state is not None:
                    print(f"\n🛟 MAX ITERATIONS REACHED. Restoring Best Incumbent Design...")
                    break
                else:
                    print(f"\n🛟 SAFE-SWING FALLBACK | FORCING UPPER BOUND: {v_ceiling*100:.1f}%")
                    v_mid = v_ceiling
                
            print(f"\n========================================================")
            print(f" 🌐 OUTER LOOP {outer_iter+1} | LOCKING VOLUME TARGET: {v_mid*100:.1f}%")
            print(f"========================================================")
            
            # Reset Gamma to user target at the start of every new volume baseline
            gamma = float(self.config.get('optimization_solver', {}).get('inertia_weight', 0.15))
            
            for g_iter in range(gamma_iters):
                print(f"\n   -> [Gamma Tuning {g_iter+1}/{gamma_iters}] Testing Inertia Penalty (γ): {gamma:.3f}")
                
                self.config['constraints']['target_volume_fraction'] = float(v_mid)
                if 'optimization_solver' not in self.config:
                    self.config['optimization_solver'] = {}
                self.config['optimization_solver']['inertia_weight'] = float(gamma)
                
                optimizer = HierarchicalTopologyOptimizer(self.config, self.fem_2d, self.fem_3d)
                rho_3d, inner_telemetry = optimizer.execute_pipeline()
                self.telemetry.extend(inner_telemetry)
                
                print("   -> Projecting raw topology through CNC Manufacturing Constraints...")
                ram_payload = {'densities': rho_3d, 'nodes': self.mesh_3d['nodes'], 'elements': self.mesh_3d['elements']}
                exporter = SprocketCAMExporter(ram_payload, self.config)
                
                try:
                    milling_pockets = exporter._generate_ai_pockets()
                    self.final_pockets = milling_pockets
                except RuntimeError as e:
                    if "CNC LOGIC FAILURE" in str(e):
                        print(f"   [⚠️] CAM FILTER FAILURE: Disconnected spokes detected (Topology is broken).")
                        is_failing = True
                        worst_margin = 0.5 # Set a 50% structural penalty to trigger a smooth 15% volume jump later
                        
                        if gamma > 0.01 and g_iter < gamma_iters - 1:
                            print(f"      -> ⚙️ DIAGNOSIS: Reducing Inertia Penalty to pull mass back into the structural web.")
                            gamma = max(0.01, gamma * 0.5)
                            print(f"      -> ⚙️ ACTIVE-SET: Gamma forced down to {gamma:.3f}")
                            continue # Skip Salome, immediately rerun topology at same volume but lower gamma
                        else:
                            print(f"      -> ⚙️ DIAGNOSIS: Gamma floor exhausted. Global mass budget is too low.")
                            self.optimal_densities = np.copy(rho_3d)
                            self.final_stresses = np.zeros(len(self.mesh_3d['nodes']))
                            self.final_deflections = np.zeros_like(self.mesh_3d['nodes'])
                            self.final_v_mid = v_mid
                            self.final_void_count = 0
                            break # Break Inner Loop to trigger Outer Loop volume addition
                    else:
                        raise e
                        
                prep_pockets = prep(milling_pockets)
                audited_rho = np.copy(rho_3d)
                nodes_3d = self.mesh_3d['nodes'] * 1000.0 
                elements_3d = self.mesh_3d['elements']
                bounds_min = self.mesh_3d['bounds_min']
                elem_nodes = nodes_3d[elements_3d]
                centroids_mm = np.mean(elem_nodes, axis=1)[:, :2]
                
                void_count = 0
                for i in range(len(elements_3d)):
                    if bounds_min[i] >= 1.0: continue
                    pt = Point(centroids_mm[i, 0], centroids_mm[i, 1])
                    if prep_pockets.contains(pt):
                        audited_rho[i] = 0.01
                        void_count += 1
                        
                self.optimal_densities = audited_rho
                
                print("   -> Exporting mid-loop CAD geometry for Salome validation...")
                ram_payload = {'densities': audited_rho, 'nodes': self.mesh_3d['nodes'], 'elements': self.mesh_3d['elements']}
                exporter = SprocketCAMExporter(ram_payload, self.config)
                step_path = os.path.join(self.run_dir, "PRAVAHA_Sprocket_Optimized.step")
                exporter.export_3d_step(self.final_pockets, step_path)
                
                print("   -> [Salome Judge] Offloading to Code_Aster for Direct Matrix Solve...")
                safe_payload = {}
                for k, v in self.payload.items():
                    if isinstance(v, np.ndarray): safe_payload[k] = v.tolist()
                    else: safe_payload[k] = v
                        
                salome_cfg = {"config": self.config, "payload": safe_payload}
                with open(os.path.join(self.run_dir, "salome_config.json"), "w") as f:
                    json.dump(salome_cfg, f)
                    
                src_dir = os.path.dirname(os.path.abspath(__file__))
                launcher_path = "/tmp/pravaha_salome_launcher.py"
                with open(launcher_path, "w") as f:
                    f.write(f'''
import sys
sys.path.insert(0, r"{src_dir}")
import salome_validator
validator = salome_validator.SalomeSprocketValidator(r"{self.run_dir}")
validator.process_cad_geometry()
validator.generate_quadratic_mesh()
validator.write_code_aster_comm()
validator.run_aster_solver()
validator.parse_results_and_report_fos()
''')
                cmd = ["salome", "-t", "python", launcher_path]
                try:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/tmp", bufsize=1)
                    for line in iter(process.stdout.readline, ''):
                        if line.strip(): print(f"      [Salome] {line.strip()}")
                    process.wait()
                    if process.returncode != 0:
                        print(f"   [⚠️] Salome Validation crashed. Halting pipeline.")
                        import sys; sys.exit(1)
                except Exception as e:
                    print(f"   [⚠️] Salome Execution failed: {str(e)}")
                    import sys; sys.exit(1)
                finally:
                    if os.path.exists(launcher_path): os.remove(launcher_path)
                    
                results_file = os.path.join(self.run_dir, "salome_results.json")
                with open(results_file, 'r') as f:
                    salome_data = json.load(f)
                    
                fea_fos_web = float(salome_data.get('fos', 1.0))
                # THE FIX: Explicitly fetch the new isolated tooth FoS key from Salome
                fea_fos_tooth = float(salome_data.get('fos_tooth', salome_data.get('fos', 1.0)))
                fea_fos_shear = float(salome_data.get('fos_shear', 1.0))
                in_plane_def = float(salome_data.get('max_in_plane_deflection_m', 1.0))
                axial_def = float(salome_data.get('max_axial_deflection_m', 1.0))
                
                print("   -> Interpolating Code_Aster stress field onto SIMP grid for ML...")
                try:
                    pc_path = os.path.join(self.run_dir, "salome_point_cloud.npz")
                    pc_data = np.load(pc_path)
                    salome_coords = pc_data['coords']
                    salome_vmis = pc_data['vmis']

                    if len(salome_coords) > 1:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(salome_coords)
                        simp_nodes_mm = self.mesh_3d['nodes'] * 1000.0
                        distances, indices = tree.query(simp_nodes_mm)
                        self.final_stresses = salome_vmis[indices].flatten() * 1e6
                        self.final_deflections = np.zeros_like(self.mesh_3d['nodes']) 
                    else:
                        raise ValueError("Point cloud array was empty.")
                except Exception as e:
                    print(f"   [⚠️] Spatial interpolation failed: {str(e)}")
                    self.final_stresses = np.zeros(len(self.mesh_3d['nodes']))
                    self.final_deflections = np.zeros_like(self.mesh_3d['nodes'])
                
                analytical_fos_hub = self.evaluate_analytical_fos(v_mid)
                analytical_fos_tooth = self.evaluate_analytical_tooth_failsafe()
                driving_fos_web = min(fea_fos_web, analytical_fos_hub)
                driving_fos_tooth = min(fea_fos_tooth, analytical_fos_tooth)
                
                self.run_metadata['fea_fos_web'] = float(fea_fos_web)
                self.run_metadata['fea_fos_shear'] = float(fea_fos_shear)
                self.run_metadata['fea_in_plane_deflection_m'] = float(in_plane_def)
                self.run_metadata['fea_axial_deflection_m'] = float(axial_def)
                self.run_metadata['fea_fos_tooth'] = float(fea_fos_tooth)
                self.run_metadata['analytical_fos_tooth'] = float(analytical_fos_tooth)
                self.run_metadata['final_inertia_penalty_gamma'] = float(gamma)
                self.final_v_mid = v_mid
                self.final_void_count = void_count
                
                print(f"\n   -> 3D FEA FoS [TOOTH]       : {fea_fos_tooth:.2f} (Floor: {minimum_tooth_fos:.2f})")
                print(f"   -> 3D FEA FoS [WEB VMIS]    : {fea_fos_web:.2f} (Target: {target_web_fos:.2f})")
                print(f"   -> 3D FEA FoS [WEB SHEAR]   : {fea_fos_shear:.2f} (Target: {target_shear_fos:.2f})")
                print(f"   -> Kinematics [IN-PLANE]    : {in_plane_def*1000:.4f} mm (Limit: {max_in_plane*1000:.4f} mm)")
                print(f"   -> Kinematics [AXIAL]       : {axial_def*1000:.4f} mm (Limit: {max_axial*1000:.4f} mm)")
                
                margin_tooth = driving_fos_tooth / minimum_tooth_fos
                margin_vmis = driving_fos_web / target_web_fos
                margin_shear = fea_fos_shear / target_shear_fos
                margin_in_plane = max_in_plane / in_plane_def if in_plane_def > 0 else 999.0
                margin_axial = max_axial / axial_def if axial_def > 0 else 999.0

                margin_kinematic = min(margin_in_plane, margin_axial)
                margin_topology_stress = min(margin_vmis, margin_shear)
                
                is_kinematic_failing = margin_kinematic < 1.0
                is_stress_failing = margin_topology_stress < 1.0
                is_tooth_failing = margin_tooth < 1.0
                
                is_failing = is_kinematic_failing or is_stress_failing or is_tooth_failing
                
                # The FSD Volume Solver scales based on the absolute worst margin, including teeth
                worst_margin = min(margin_kinematic, margin_topology_stress, margin_tooth)
                
                self.ml_sample_counter += 1
                
                # Create an objective filename containing the driving input parameters
                ml_filename = os.path.join(self.ml_dir, f"sample_{self.ml_sample_counter:03d}_V{int(v_mid*100)}_G{int(gamma*100)}.npz")
                
                # Calculate inertia just for the ML dataset tracking
                current_inertia = self._calculate_rotational_inertia(audited_rho)
                
                np.savez_compressed(
                    ml_filename,
                    densities=audited_rho,                 # ML Input Feature: 3D Geometry Array
                    stresses=self.final_stresses,          # ML Target: Raw 3D Von Mises Stress Tensor (Pa)
                    deflections=self.final_deflections,    # ML Target: Raw 3D Deflection Tensor (m)
                    v_mid=v_mid,                           # ML Global Feature: Target Volume Fraction
                    gamma=gamma,                           # ML Global Feature: Inertia Penalty
                    inertia_kgm2=current_inertia,          # ML Global Target: Physical Inertia
                    in_plane_def_m=in_plane_def,           # ML Global Target: Max In-Plane Deflection (m)
                    axial_def_m=axial_def                  # ML Global Target: Max Axial Deflection (m)
                )
                print(f"      -> 🧠 ML Harvester: Saved objective physical state {self.ml_sample_counter:03d} to dataset.")

                if not is_failing:
                    print(f"\n   ✅ Local Pareto Search Succeeded! Constraints passed at {v_mid*100:.1f}% Vol, Gamma={gamma:.3f}")
                    
                    current_inertia = self._calculate_rotational_inertia(audited_rho)
                    print(f"      -> ⚖️ Rotational Inertia (Iz): {current_inertia:.4e} kg*m^2")
                    
                    state_dict = {
                        'v_mid': v_mid,
                        'gamma': gamma,
                        'inertia': current_inertia,
                        'void_count': void_count,
                        'run_metadata': self.run_metadata.copy()
                    }
                    self.feasible_vault.append(state_dict)
                    
                    # 1. Save the specific checkpoint array (.npz)
                    file_suffix = f"V{int(v_mid*100)}_G{int(gamma*100)}_Iz_{current_inertia:.2e}"
                    temp_npz = os.path.join(self.run_dir, f"topology_3d_{file_suffix}.npz")
                    np.savez_compressed(temp_npz, densities=audited_rho, nodes=self.mesh_3d['nodes'], elements=self.mesh_3d['elements'], stresses=self.final_stresses, deflections=self.final_deflections)
                    print(f"      -> 💾 Archived valid design array to: {os.path.basename(temp_npz)}")
                    
                    # --- Save the specific checkpoint telemetry (.json) ---
                    interim_builder = AcademicPassportBuilder(
                        orchestrator=self, 
                        total_runtime_sec=0.0, # Interim passes don't log total pipeline time
                        final_v_mid=v_mid, 
                        void_count=void_count
                    )
                    interim_builder.meta = self.run_metadata.copy() # Lock in current metadata
                    interim_builder.generate_and_save(self.run_dir, filename=f"digital_twin_{file_suffix}.json")
                    
                    # --- ELITISM: IS THIS THE ABSOLUTE BEST DESIGN YET? ---
                    if current_inertia < self.best_feasible_inertia:
                        self.best_feasible_inertia = current_inertia
                        self.best_feasible_state = {
                            'densities': np.copy(audited_rho),
                            'pockets': self.final_pockets,
                            'stresses': np.copy(self.final_stresses),
                            'deflections': np.copy(self.final_deflections),
                            'v_mid': v_mid,
                            'gamma': gamma,
                            'inertia': current_inertia, # Store the inertia
                            'void_count': void_count,
                            'run_metadata': self.run_metadata.copy()
                        }
                        print(f"      -> 🏆 NEW INCUMBENT: Absolute lowest Rotational Inertia achieved so far!")
                        
                    break # Break the inner gamma loop. The volume is successfully proven!
                    
                # If we are failing, analyze if we can fix it by tuning gamma WITHOUT adding volume
                failed_reasons = []
                if margin_vmis < 1.0: failed_reasons.append(f"Web VMIS")
                if margin_shear < 1.0: failed_reasons.append(f"Web Shear")
                if margin_in_plane < 1.0: failed_reasons.append(f"In-Plane Deflect")
                if margin_axial < 1.0: failed_reasons.append(f"Axial Deflect")
                
                if margin_tooth < 1.0:
                    print(f"\n   [⚠️] WARNING: Tooth FoS is critically low ({margin_tooth:.2f}x). Relying on added web volume to stiffen roots.")
                    
                if is_failing:
                    print(f"\n   ❌ Local Pareto Search Failed: [{', '.join(failed_reasons)}]")
                
                if gamma > 0.01:
                    # --- THE FIX: PROPORTIONAL GAMMA TUNING WITH MOVE LIMITS ---
                    # Scale gamma smoothly based on the exact severity of the failure (worst_margin)
                    
                    if is_kinematic_failing:
                        print(f"      -> ⚙️ DIAGNOSIS: Kinematic Failure (Margin: {worst_margin:.3f}x)")
                        # Kinematic failures require stiffer rims. Apply an accelerated proportional drop.
                        gamma_scale = worst_margin ** 1.5 
                        # SAO Move Limit: Do not allow gamma to drop by more than 66% in a single step
                        gamma_scale = max(0.33, gamma_scale)
                    else:
                        print(f"      -> ⚙️ DIAGNOSIS: Stress Failure (Margin: {worst_margin:.3f}x)")
                        # Pure proportional response
                        gamma_scale = worst_margin
                        # SAO Move Limit: Do not allow gamma to drop by more than 50% in a single step
                        gamma_scale = max(0.50, gamma_scale)
                        
                    # Apply the scaled multiplier and enforce the absolute floor
                    gamma = max(0.01, gamma * gamma_scale)
                    print(f"      -> ⚙️ ACTIVE-SET: Proportional Gamma scaled down by {gamma_scale:.2f}x to {gamma:.3f}")
                        
                    if g_iter < gamma_iters - 1:
                        continue # Re-run the inner loop at the SAME volume fraction
                
                print(f"      -> ⚙️ DIAGNOSIS: Pareto tuning exhausted. Mass budget is structurally insufficient.")
                break # Break inner loop. We MUST increase volume in the outer loop.
            
            # --- THE FIX: UNBOUNDED EXPLORATORY FSD ---
            if outer_iter == max_iterations: break
            
            # --- THE FIX: BIDIRECTIONAL REVERSAL LOGIC ---
            if is_failing:
                is_stalled = False
                if self.last_direction == "UP" and outer_iter > 0:
                    # 1. Did the overall worst margin fail to meaningfully improve?
                    if worst_margin <= self.previous_margin + plateau_min_improvement:
                        is_stalled = True
                    # 2. Did the Tooth FoS actively degrade while adding mass? (Buffered by 0.005 to prevent numerical noise)
                    elif margin_tooth < (self.previous_margin_tooth - 0.005) and margin_tooth < 1.0:
                        print(f"   ⚠️ ANOMALY: Tooth FoS degraded ({(margin_tooth - self.previous_margin_tooth):.3f}x) despite added mass!")
                        is_stalled = True
                    # 3. Did the Web FoS actively degrade while adding mass?
                    elif margin_topology_stress < (self.previous_margin_stress - 0.005) and margin_topology_stress < 1.0:
                        print(f"   ⚠️ ANOMALY: Web FoS degraded ({(margin_topology_stress - self.previous_margin_stress):.3f}x) despite added mass!")
                        is_stalled = True
                    # 4. Did the Kinematics actively degrade while adding mass?
                    elif margin_kinematic < (self.previous_margin_kinematic - 0.005) and margin_kinematic < 1.0:
                        print(f"   ⚠️ ANOMALY: Kinematics degraded ({(margin_kinematic - self.previous_margin_kinematic):.3f}x) despite added mass!")
                        is_stalled = True

                if is_stalled:
                    self.stalled_counter += 1
                    print(f"   ⚠️ WARNING: Plateau detector triggered (Streak: {self.stalled_counter}/{max_stalled})")
                else:
                    self.stalled_counter = 0 
                    
                self.previous_margin = worst_margin
                self.previous_margin_tooth = margin_tooth
                self.previous_margin_stress = margin_topology_stress
                
                if self.stalled_counter >= max_stalled:
                    print(f"\n   🔄 LOCAL MINIMUM DETECTED: Added mass is reducing strength! Switching directions to scramble topology.")
                    self.stalled_counter = 0
                    self.last_direction = "DOWN"
                    
                    v_mid = max(v_floor, v_mid * plateau_scramble_multiplier) # Aggressive dynamic drop to shatter the stubborn topology
                    print(f"\n   ⬇️ SCRAMBLING TOPOLOGY: Stepping Volume DOWN to {v_mid*100:.1f}%")
                else:
                    self.last_direction = "UP"
                    vol_scale = (1.0 / worst_margin) ** 0.5 
                    v_next_ideal = v_mid * vol_scale
                    # THE FIX: Percentage-based limits (Current volume + 15%)
                    max_allowed_jump = v_mid * 1.15
                    v_mid = min(v_ceiling, min(max_allowed_jump, v_next_ideal))
                    
                    print(f"\n   ⬆️ EXPLORATORY FSD: Stepping Volume UP to {v_mid*100:.1f}%")
                
            else:
                lowest_margin = min(margin_kinematic, margin_topology_stress, margin_tooth)
                
                # We expect margin to drop when cutting mass. If it miraculously improved while going DOWN, 
                # or if it barely dropped at all, the topology is highly stable. No strike recorded here since it's "Safe".
                self.stalled_counter = 0 
                self.previous_margin = lowest_margin 
                self.previous_margin_tooth = margin_tooth
                self.previous_margin_stress = margin_topology_stress
                self.previous_margin_in_plane = margin_in_plane
                self.previous_margin_axial = margin_axial
                
                print(f"\n   📉 Machined Web is SAFE (Margin: {lowest_margin:.3f}x). Pushing for absolute minimum weight...")
                
                self.last_direction = "DOWN"
                vol_scale = (1.0 / lowest_margin) ** 0.5 
                v_next_ideal = v_mid * vol_scale
                max_allowed_drop = v_mid * 0.85
                v_mid = max(v_floor, max(max_allowed_drop, v_next_ideal))
                
                print(f"\n   ⬇️ EXPLORATORY FSD: Stepping Volume DOWN to {v_mid*100:.1f}%")
                
            # Convergence Check
            if abs(v_mid - self.previous_v_mid) < 0.005:
                if not is_failing:
                    print(f"   ✅ Volume stabilized at safe margin. Optimal topology found.")
                    break
                else:
                    print(f"   ⚠️ Volume locked at upper ceiling. Waiting for plateau scramble...")
            self.previous_v_mid = v_mid
                
        if self.best_feasible_state is not None:
            print(f"\n🏆 OPTIMIZATION COMPLETE. Restoring absolute best valid design!")
            print(f"   -> Optimal Volume Fraction: {self.best_feasible_state['v_mid']*100:.1f}%")
            print(f"   -> Minimum Rotational Inertia: {self.best_feasible_state['inertia']:.4e} kg*m^2")
            print(f"   -> Total Feasible Designs Archived: {len(self.feasible_vault)}")
            
            self.optimal_densities = self.best_feasible_state['densities']
            self.final_pockets = self.best_feasible_state['pockets']
            self.final_stresses = self.best_feasible_state['stresses']
            self.final_deflections = self.best_feasible_state['deflections']
            self.final_v_mid = self.best_feasible_state['v_mid']
            self.final_void_count = self.best_feasible_state['void_count']
            self.run_metadata = self.best_feasible_state['run_metadata']
            
            # Save the vault ledger to JSON for later analysis
            vault_path = os.path.join(self.run_dir, "pareto_archive_ledger.json")
            with open(vault_path, "w") as f:
                json.dump(self.feasible_vault, f, indent=4)
                
        elif is_failing:
            print(f"\n⚠️ WARNING: Pipeline exhausted all iterations without finding a safe design. Exporting safest failed configuration.")

    def export_artifacts(self, total_runtime_sec=0.0):
        """Phase 3: Data Archiving (Decoupled from DFM)."""
        print("\n[4/4] EXPORTING MATHEMATICAL ARTIFACTS...")
        
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Is_3D_Phase", "Iteration", "Envelope_Compliance_Nm", "Max_Delta_Rho", "Compute_Time_sec", "Peak_VRAM_GB"])
            writer.writerows(self.telemetry)
            
        # --- Pack physical tensors into the archive for ParaView ---
        np.savez_compressed(self.npz_path, 
                            densities=self.optimal_densities, 
                            nodes=self.mesh_3d['nodes'], 
                            elements=self.mesh_3d['elements'],
                            stresses=self.final_stresses,
                            deflections=self.final_deflections)
                            
        with open(self.json_path, "w") as f:
            json.dump(self.run_metadata, f, indent=4)
            
        # --- Build and Save the Deep Research Passport ---
        if hasattr(self, 'final_v_mid'):
            passport_builder = AcademicPassportBuilder(
                orchestrator=self, 
                total_runtime_sec=total_runtime_sec, 
                final_v_mid=self.final_v_mid, 
                void_count=self.final_void_count
            )
            passport_builder.generate_and_save(self.run_dir)
            
        print(f"   ✅ Artifacts successfully saved to: {self.run_dir}")
        print(f"   ➡ Ready for DFM Post-Processing. Run manufacturing_export.py manually to generate the Production STEP file.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_yaml = os.path.abspath(os.path.join(base_dir, "../config/sprocket_config.yaml"))
    materials_yaml = os.path.abspath(os.path.join(base_dir, "../config/materials_library.yaml"))
    
    total_start = time.time()
    
    try:
        orchestrator = PravahaOrchestratorV2(config_yaml, materials_yaml)
        
        # Phase 1: Procedural Generation & Physics Boundary setup
        orchestrator.execute_cold_path()
        
        # Phase 2: Decoupled Multi-Constraint & Pareto Frontier Outer Loop
        orchestrator.execute_outer_loop()
        
        # Phase 3: Artifact Archiving & ML Tensor Mapping
        runtime_sec = time.time() - total_start
        orchestrator.export_artifacts(total_runtime_sec=runtime_sec)
        
        runtime_min = (time.time() - total_start) / 60.0
        print("\n" + "="*70)
        print(f" 🏁 PRAVAHA V2 COMPUTE COMPLETED IN {runtime_min:.2f} MINUTES")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ FATAL PIPELINE CRASH: {str(e)}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

