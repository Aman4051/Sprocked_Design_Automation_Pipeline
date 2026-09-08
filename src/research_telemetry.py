import os
import json
import numpy as np
from datetime import datetime

class AcademicPassportBuilder:
    """
    Decoupled Analytics Engine for Research Publication.
    Extracts 3D spatial geometry, deterministic FoS, and kinematic 
    boundaries to generate a 'Digital Twin Passport' JSON.
    Updated for the ASME B29.1 Architecture.
    """
    def __init__(self, orchestrator, total_runtime_sec, final_v_mid, void_count):
        self.orch = orchestrator
        self.config = orchestrator.config
        self.payload = orchestrator.payload
        self.mesh = orchestrator.mesh_3d
        self.meta = orchestrator.run_metadata
        
        self.runtime_sec = total_runtime_sec
        self.v_mid = final_v_mid
        self.void_count = void_count

    # --- THE FIX: Add filename parameter ---
    def generate_and_save(self, output_dir, filename="digital_twin_passport.json"):
        print(f"   -> [Telemetry] Compiling Academic Digital Twin Passport...")
        
        # 1. Mesh & Geometric Analytics
        num_nodes = len(self.mesh['nodes'])
        num_elements = len(self.mesh['elements'])
        
        # Calculate Base Mass vs Optimized Mass
        rho_mat = float(self.payload.get('density', 2810.0))
        web_thick = float(self.config.get('geometry', {}).get('web_thickness_m', 0.007))
        r_pitch = float(self.payload.get('R_pitch', 0.1))
        
        solid_vol = np.pi * (r_pitch**2) * web_thick
        solid_mass_kg = solid_vol * rho_mat
        optimized_mass_kg = solid_mass_kg * self.v_mid
        
        # 2. Extract Dictionaries safely
        geom_cfg = self.config.get('geometry', {})
        kin_cfg = self.config.get('kinematics', {})
        pwr_cfg = self.config.get('powerplant', {})
        opt_cfg = self.config.get('constraints', {})
        
        # 3. Build the Deep JSON Structure
        passport = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "solver_engine": "PRAVAHA_V2_3D_Continuum_GPU",
                "optimization_strategy": "Bisection with CNC Heaviside Projection",
                "target_volume_fraction": opt_cfg.get('target_volume_fraction')
            },
            "sprocket_architecture": {
                "asme_b29_1_type": geom_cfg.get('sprocket_type', 'C'),
                "pitch_inch": geom_cfg.get('pitch_inch', 0.5),
                "num_teeth": geom_cfg.get('num_teeth', 39),
                "num_bolts": geom_cfg.get('num_bolts', 4),
                "bolt_pcd_m": geom_cfg.get('bolt_pcd_m', 0.081)
            },
            "physics_boundary_conditions": {
                "chain_tension_peak_n": float(self.payload.get('T_max', 0.0)),
                "engaged_teeth_count": int(self.payload.get('N_engaged', 0)),
                "chain_velocity_m_s": float(self.payload.get('v_chain', 0.0)),
                "derated_yield_strength_pa": float(self.payload.get('Sy_derated', 0.0)),
                "derated_youngs_modulus_pa": float(self.payload.get('E_derated', 0.0))
            },
            "computational_mesh_metrics": {
                "total_nodes": num_nodes,
                "total_tetrahedral_elements": num_elements,
                "cnc_machining_projection": {
                    "snapped_void_voxels": self.void_count,
                    "percentage_voxels_machined_away": float((self.void_count / max(1, num_elements)) * 100.0)
                }
            },
            "geometric_mass_metrics": {
                "solid_baseline_mass_kg": float(solid_mass_kg),
                "optimized_machined_mass_kg": float(optimized_mass_kg),
                "true_converged_volume_fraction": float(self.v_mid)
            },
            "structural_integrity_audit": {
                "certified_factors_of_safety": {
                    "web_3d_continuum_fos": self.meta.get('fea_fos_web', 0.0),
                    "tooth_3d_continuum_fos": self.meta.get('fea_fos_tooth', 0.0),
                    "tooth_analytical_lewis_fos": self.meta.get('analytical_fos_tooth', 0.0)
                }
            },
            "computational_cost": {
                "total_wall_clock_time_seconds": float(self.runtime_sec)
            }
        }
        
        # --- THE FIX: Use the custom filename ---
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(passport, f, indent=4)
            
        print(f"   ✅ Academic Passport saved: {os.path.basename(filepath)}")
        return filepath