import math
import numpy as np

class UnifiedPhysicsEngine:
    """
    The Single Source of Truth for PRAVAHA V2 Physics.
    Generates consistent Kinetic boundaries for both 2D Planar and 3D Spatial Continuum Solvers.
    Thermal degradation and Southwell stiffening removed; relies strictly on user-derated material inputs.
    """
    def __init__(self, config, materials_db):
        print("   -> Initializing Unified Multi-Physics Engine...")
        self.config = config
        self.materials = materials_db.get('materials', {})
        
        # Operational Baseline
        self.rpm_motor = float(config.get('max_motor_rpm', 6000))
        self.T_base = float(config.get('chain_tension_n', 650.0))
        self.pitch_m = float(config.get('pitch_inch', 0.5)) * 0.0254
        self.N_driven = int(config.get('num_teeth', 39))
        self.N_driver = int(config.get('other_sprocket_teeth', 13))
        self.center_dist = float(config.get('manual_center_distance_m', 0.402))
        
        # Extracted Material Keys
        self.alloy_key = config.get('active_material', 'al_7075_t6')
        
    def extract_material_properties(self):
        """Pulls user-provided (pre-derated) properties directly from the DB."""
        mat = self.materials.get(self.alloy_key, {})
        
        # User inputs pre-degraded values directly into their database/config
        E_user = mat.get('youngs_modulus', 70e9)
        Sy_user = mat.get('yield_strength', 503e6)
        nu = mat.get('poissons_ratio', 0.33)
        rho = mat.get('density', 2810.0)
        
        print(f"      -> Material [{self.alloy_key}]: Using user-supplied properties: E = {E_user/1e9:.1f} GPa | Sy = {Sy_user/1e6:.1f} MPa")
        
        return E_user, Sy_user, nu, rho

    def evaluate_kinematics_and_tension(self):
        """Calculates true dynamic engagement and tension using precise B29.1 physics."""
        R_driven = self.pitch_m / (2.0 * math.sin(math.pi / self.N_driven))
        R_driver = self.pitch_m / (2.0 * math.sin(math.pi / self.N_driver))
        
        # 1. Exact Catenary Sag & Wrap Angle Deviation 
        beta_rad = math.asin(abs(R_driven - R_driver) / self.center_dist)
        L_span = self.center_dist * math.cos(beta_rad)
        
        slack_percentage = self.config.get('chain_slack_percentage', 3.0) / 100.0
        E_slack = L_span * slack_percentage 
        
        S_sag = (3.0 * L_span * E_slack) / 8.0
        phi_rad = math.atan((4.0 * S_sag) / L_span)
        
        # Case A: Slack Span on the Bottom
        if R_driven > R_driver:
            theta_actual = math.pi + (2.0 * beta_rad) + phi_rad
        else:
            theta_actual = math.pi - (2.0 * beta_rad) + phi_rad
            
        gamma_pitch_rad = (2.0 * math.pi) / self.N_driven
        N_engaged = int(math.floor(theta_actual / gamma_pitch_rad)) + 1
        N_engaged = max(2, N_engaged)
        
        # 2. True Kinematic Velocity 
        v_chain = (self.N_driver * self.pitch_m * self.rpm_motor) / 60.0
        
        # 3. Dynamic Tension Components (Centrifugal omitted for sprocket analysis)
        v_impact = ((self.pitch_m * math.pi * self.rpm_motor) / 30.0) * math.sin(math.pi / self.N_driven)
        
        # --- THE FIX: Extract chain kinetic properties dynamically from config ---
        kin_cfg = self.config.get('kinematics', {})
        k_chain = float(kin_cfg.get('chain_stiffness_n_m', 500000.0))
        m_link = float(kin_cfg.get('chain_link_mass_kg', 0.015))
        
        F_shock = v_impact * math.sqrt(k_chain * m_link)
        
        # Operational Service Multiplier
        pp = self.config.get('powerplant', {})
        service_factor = pp.get('service_factor', 1.5)
        
        # Final Max Tension
        T_max = (service_factor * self.T_base) + F_shock 
        
        print(f"      -> Kinematics: V_chain = {v_chain:.2f} m/s | Engaged Teeth: {N_engaged} | T_max = {T_max:.1f} N")
        return R_driven, N_engaged, T_max, v_chain

    def generate_dual_load_envelope(self, N_engaged, T_max):
        """Creates both 2D and 3D spatial vectors for BOL (Perfect Pitch) and EOL (Stretched)."""
        pitch_angle_rad = math.radians(360.0 / self.N_driven)
        
        # Average pressure angle from documentation
        pressure_angle_deg = 26.0 - (92.0 / self.N_driven)
        theta = math.radians(pressure_angle_deg)
        
        # Exact Link and Tooth force transfer decay formulas 
        decay_factor = math.sin(theta) / (math.sin(pitch_angle_rad) + math.sin(theta))
        tooth_force_factor = math.sin(pitch_angle_rad) / (math.sin(pitch_angle_rad) + math.sin(theta))
        
        F_planar_BOL = np.zeros((self.N_driven, 2))
        F_planar_EOL = np.zeros((self.N_driven, 2))
        
        F_spatial_BOL = np.zeros((self.N_driven, 3))
        F_spatial_EOL = np.zeros((self.N_driven, 3))
        
        z_slap_angle = math.radians(self.config.get('chain_misalignment_deg', 2.0))
        
        for i in range(1, N_engaged + 1):
            # --- THE FIX: Geometrical Phase Alignment ---
            # Shift by 90 degrees (pi/2) to synchronize the physical load vectors 
            # with the procedural CAD generator which starts its first gap on the Y-axis.
            theta_tooth = (i - 1) * pitch_angle_rad + (math.pi / 2.0)
            
            # Case 1: BOL (Decaying from Entry)
            T_in_BOL = T_max * (decay_factor ** (i - 1))
            Fn_BOL = T_in_BOL * tooth_force_factor
            
            # Case 2: EOL (Accumulating to Exit)
            T_in_EOL = T_max * (decay_factor ** (N_engaged - i))
            Fn_EOL = T_in_EOL * tooth_force_factor
            
            # INWARD CRUSHING VECTORS (Fx = -Fn*sin, Fy = Fn*cos)
            vec_x_BOL = -Fn_BOL * math.sin(theta_tooth + theta)
            vec_y_BOL = Fn_BOL * math.cos(theta_tooth + theta)
            
            vec_x_EOL = -Fn_EOL * math.sin(theta_tooth + theta)
            vec_y_EOL = Fn_EOL * math.cos(theta_tooth + theta)
            
            # Populate 2D Planar
            F_planar_BOL[i-1, :] = [vec_x_BOL, vec_y_BOL]
            F_planar_EOL[i-1, :] = [vec_x_EOL, vec_y_EOL]
            
            # Populate 3D Spatial (Extruding Z-Slap)
            F_spatial_BOL[i-1, :] = [vec_x_BOL, vec_y_BOL, Fn_BOL * math.sin(z_slap_angle)]
            F_spatial_EOL[i-1, :] = [vec_x_EOL, vec_y_EOL, Fn_EOL * math.sin(z_slap_angle)]
            
        print(f"      -> Dual-Case Envelope: Generated 2D and 3D Boundary Vectors.")
        return F_planar_BOL, F_planar_EOL, F_spatial_BOL, F_spatial_EOL

    def build_physics_payload(self):
        """Executes all constraints and returns the unified dictionary."""
        print("\n[Unified Physics Engine] Compiling Topological Boundary Parameters...")
        
        R_pitch, N_eng, T_max, v_chain = self.evaluate_kinematics_and_tension()
        E_user, Sy_user, nu, rho = self.extract_material_properties()
        
        F_2D_BOL, F_2D_EOL, F_3D_BOL, F_3D_EOL = self.generate_dual_load_envelope(N_eng, T_max)
        
        payload = {
            'R_pitch': R_pitch,
            'N_engaged': N_eng,
            'T_max': T_max,
            'v_chain': v_chain,
            'E_derated': E_user,
            'Sy_derated': Sy_user,
            'nu': nu,
            'density': rho,
            'F_planar_BOL': F_2D_BOL,
            'F_planar_EOL': F_2D_EOL,
            'F_spatial_BOL': F_3D_BOL,
            'F_spatial_EOL': F_3D_EOL
        }
        
        print("[Unified Physics Engine] Payload Assembled. Ready for Dual-Mesher.\n")
        return payload