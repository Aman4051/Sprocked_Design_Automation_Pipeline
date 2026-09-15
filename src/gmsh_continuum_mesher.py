import math
import numpy as np
import gmsh
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
import cadquery as cq
import os
import tempfile

class GmshContinuumMesher3D:
    def __init__(self, config, physics_payload, profile_coords, gap_coords, is_coarse=False):
        print(f"   -> Initializing Unified Dual-Continuum Mesher...")
        self.config = config
        self.payload = physics_payload
        self.profile_coords = profile_coords
        self.gap_coords = gap_coords 
        self.is_coarse = is_coarse
        
        # --- Extract Sprocket Type ---
        self.sprocket_type = str(self.config.get('geometry', {}).get('sprocket_type', 'C')).upper()
        
        # Base Dimensions
        self.bolt_pcd_m = float(self.config.get('geometry', {}).get('bolt_pcd_m', 0.081))
        self.num_bolts = int(self.config.get('geometry', {}).get('num_bolts', 4))
        self.bolt_dia_m = float(self.config.get('geometry', {}).get('bolt_dia_m', 0.008))
        self.thickness_m = float(self.config.get('geometry', {}).get('web_thickness_m', 0.007))
        self.roller_rad_m = (float(self.config.get('kinematics', {}).get('roller_dia_inch', 0.335)) * 0.0254) / 2.0
        
        self.R_pitch = float(self.payload['R_pitch'])
        self.N_driven = int(self.config.get('geometry', {}).get('num_teeth', 39))
        self.N_engaged = int(self.payload['N_engaged'])
        self.mesh_size = float(self.config.get('continuum_meshing_3d', {}).get('starting_voxel_size_m', 0.003))
        self.pitch_m = float(self.config.get('geometry', {}).get('pitch_inch', 0.5)) * 0.0254
        
        # Pull both 2D and 3D forces from unified_physics.py
        self.forces_BOL_2D = self.payload.get('F_planar_BOL', np.zeros((self.N_engaged, 2)))
        self.forces_EOL_2D = self.payload.get('F_planar_EOL', np.zeros((self.N_engaged, 2)))
        self.forces_BOL_3D = self.payload.get('F_spatial_BOL', np.zeros((self.N_engaged, 3)))
        self.forces_EOL_3D = self.payload.get('F_spatial_EOL', np.zeros((self.N_engaged, 3)))
        
        # Constraint limits (Lug Shear)
        torque = self.payload['T_max'] * self.R_pitch
        F_bolt_max = (torque / (self.bolt_pcd_m / 2.0)) / max(2, self.num_bolts / 2.0) if self.num_bolts > 0 else 0
        Sy = float(self.payload['Sy_derated'])
        S_sy = 0.577 * Sy
        FS = float(self.config.get('constraints', {}).get('minimum_allowed_factor_of_safety', 1.2))
        
        if self.num_bolts > 0:
            e_req = (F_bolt_max * FS * 3.0) / (2.0 * self.thickness_m * S_sy)
            self.boss_rad_m = max((self.bolt_dia_m / 2.0) + e_req, self.bolt_dia_m)
        else:
            self.boss_rad_m = 0.0
            
        # Prioritize YAML user input for Bore Diameter over implicit calculation
        self.bore_rad_m = float(self.config.get('geometry', {}).get('bore_dia_m', (self.bolt_pcd_m - (2*self.boss_rad_m)))) / 2.0

        # --- Calculate ASME B29.1 Hub Radius in Meters ---
        PD_m = self.pitch_m / math.sin(math.pi / self.N_driven)
        H_m = self.config.get('geometry', {}).get('hub_projection_m')
        if H_m is None:
            H_inch = 0.375 + ((self.bore_rad_m * 2.0)/0.0254 / 6.0) + 0.01 * (PD_m/0.0254)
            H_m = H_inch * 0.0254
            
        self.hub_rad_m = self.bore_rad_m + H_m
        
        # Failsafe: Hub cannot intersect tooth roots
        Root_Dia_m = (self.R_pitch * 2.0) - (self.roller_rad_m * 2.0)
        if (self.hub_rad_m * 2.0) > Root_Dia_m - 0.005:
            self.hub_rad_m = (Root_Dia_m - 0.005) / 2.0

        # We need Z-bounds for the FEA Boundary mapping later
        self._calculate_z_bounds()

    def _calculate_z_bounds(self):
        """
        Calculates the spatial Z-boundaries of the sprocket for 3D meshing and load mapping.
        Accounts for different ANSI B29.1 sprocket configurations (Types A, B, C, D).
        """
        # Fetch or calculate the hub projection (H_m)
        PD_m = self.pitch_m / math.sin(math.pi / self.N_driven)
        H_m = self.config.get('geometry', {}).get('hub_projection_m')
        
        if H_m is None:
            # Standard ASME B29.1 empirical hub length formula
            H_inch = 0.375 + ((self.bore_rad_m * 2.0) / 0.0254 / 6.0) + 0.01 * (PD_m / 0.0254)
            H_m = H_inch * 0.0254
            
        if self.sprocket_type in ['A', 'D']:
            # Flat plate (Type A) or Lug-Centric (Type D)
            self.total_z_min = -self.thickness_m / 2.0
            self.total_z_max = self.thickness_m / 2.0
            
        elif self.sprocket_type == 'B':
            # Hub projecting from one side only
            self.total_z_min = -self.thickness_m / 2.0
            self.total_z_max = (self.thickness_m / 2.0) + H_m
            
        elif self.sprocket_type == 'C':
            # Hub projecting symmetrically from both sides
            self.total_z_min = -(self.thickness_m / 2.0) - (H_m / 2.0)
            self.total_z_max = (self.thickness_m / 2.0) + (H_m / 2.0)
            
        else:
            # Fallback
            self.total_z_min = -self.thickness_m / 2.0
            self.total_z_max = self.thickness_m / 2.0
            
    def _generate_cadquery_step(self):
        """
        Uses CadQuery to generate an absolute precision B29.1 solid directly in METERS.
        Bypasses Gmsh boolean splines, eliminating BOPAlgo crashes.
        """
        print(f"      -> Assembling high-precision CadQuery solid for Type {self.sprocket_type}...")
        
        N = self.N_driven
        P_m = self.pitch_m
        Dr_m = self.roller_rad_m * 2.0 
        W_m = float(self.config.get('geometry', {}).get('chain_width_inch', 0.312)) * 0.0254
        Bore_m = self.bore_rad_m * 2.0 
        C_m = 0.0381 * 0.0254 
        
        PCD = P_m / math.sin(math.pi / N)
        t1 = self.thickness_m 
        Rf = 0.04 * P_m 
        Hub_L_m = self.total_z_max - self.total_z_min
        
        # --- THE FIX: ASME Flat-Web Profile Logic ---
        # Calculate strict ASME tooth width
        Tooth_W_m = (0.93 * (W_m / 0.0254) - 0.006) * 0.0254
        if Tooth_W_m > t1: 
            Tooth_W_m = t1 # Tooth cannot physically be thicker than the web blank
            
        Z_web = t1 / 2.0
        Z_tooth = Tooth_W_m / 2.0
        Z_tip_flat = Z_tooth / 3.0 # Flat at the very tip
        
        X_tip = P_m * (0.6 + 1.0 / math.tan(math.pi / N)) / 2.0
        X_chamfer = X_tip - (0.5 * P_m)
        X_pitch = PCD / 2.0
        X_relief_top = X_pitch - (0.2 * P_m)
        X_relief_bot = X_relief_top - (0.5 * P_m)
        
        # Ensure the bevel doesn't crash into the hub
        if self.sprocket_type in ['B', 'C']:
            X_relief_bot = max(self.hub_rad_m + Rf + 0.001, X_relief_bot)
        else:
            X_relief_bot = max(Bore_m/2.0 + 0.001, X_relief_bot)

        # Gap / Tooth Seating Profile Calculations
        A = math.radians(35.0 + 60.0 / N)
        B = math.radians(18.0 - 56.0 / N)
        M, T = 0.8 * Dr_m * math.cos(A), 0.8 * Dr_m * math.sin(A)
        W, V = 1.4 * Dr_m * math.cos(math.pi / N), 1.4 * Dr_m * math.sin(math.pi / N)
        
        R = 0.5025 * Dr_m + C_m
        E_val = 1.3025 * Dr_m + C_m
        F = Dr_m * (0.8 * math.cos(B) + 1.4 * math.cos(math.radians(17.0 - 64.0 / N)) - 1.3025) - C_m
        
        c_cx, c_cy = -M, T
        b_cx, b_cy = W, -V
        
        def arc_midpoint(p1, p2, cx, cy, direction='CCW'):
            a1 = math.atan2(p1[1] - cy, p1[0] - cx)
            a2 = math.atan2(p2[1] - cy, p2[0] - cx)
            if direction == 'CCW' and a2 < a1: a2 += 2 * math.pi
            elif direction == 'CW' and a2 > a1: a2 -= 2 * math.pi
            amid = (a1 + a2) / 2
            r = math.hypot(p1[0] - cx, p1[1] - cy)
            return (cx + r * math.cos(amid), cy + r * math.sin(amid))
            
        def get_tip_intersection():
            sin_n = math.sin(math.pi / N)
            cos_n = math.cos(math.pi / N)
            C1 = b_cx * sin_n + (PCD/2.0 + b_cy) * cos_n
            C2 = b_cx**2 + (PCD/2.0 + b_cy)**2 - F**2
            desc = C1**2 - C2
            if desc < 0:
                tooth_angle = math.pi/2 - math.pi/N
                return (b_cx + F * math.cos(tooth_angle + math.pi/2), b_cy + F * math.sin(tooth_angle + math.pi/2))
            t = C1 + math.sqrt(desc)
            return (t * sin_n, -PCD/2.0 + t * cos_n)
            
        P_bottom = (0, -R)
        P_x = (R * math.cos(-A), R * math.sin(-A))
        cy_angle = -A + B
        P_y = (c_cx + E_val * math.cos(cy_angle), c_cy + E_val * math.sin(cy_angle))
        bz_angle = math.pi + cy_angle 
        P_z = (b_cx + F * math.cos(bz_angle), b_cy + F * math.sin(bz_angle))
        P_tip = get_tip_intersection()
        pointed_radius = math.hypot(P_tip[0], P_tip[1] + PCD/2.0)
        
        P_mid_seat = arc_midpoint(P_bottom, P_x, 0, 0, 'CCW')
        P_mid_work = arc_midpoint(P_x, P_y, c_cx, c_cy, 'CCW')
        P_mid_top = arc_midpoint(P_z, P_tip, b_cx, b_cy, 'CW')
        
        dx_tip = math.sin(math.pi / N)
        dy_tip = math.cos(math.pi / N)
        P_overcut = (P_tip[0] + 0.005 * dx_tip, P_tip[1] + 0.005 * dy_tip) 
        
        wp = cq.Workplane("XZ")
        
        # --- FLAT WEB POLYGON REVOLVES ---
        if self.sprocket_type == 'A' or self.sprocket_type == 'D':
            pts = [(Bore_m/2.0, -Z_web), (X_relief_bot, -Z_web), (X_relief_top, -Z_tooth),
                   (X_chamfer, -Z_tooth), (X_tip, -Z_tip_flat), (X_tip, Z_tip_flat), 
                   (X_chamfer, Z_tooth), (X_relief_top, Z_tooth), (X_relief_bot, Z_web), 
                   (Bore_m/2.0, Z_web)]
            blank = wp.polyline(pts).close().revolve(360, (0,0,0), (0,1,0))
            
        elif self.sprocket_type == 'B':
            pts = [(Bore_m/2.0, -Z_web), (X_relief_bot, -Z_web), (X_relief_top, -Z_tooth),
                   (X_chamfer, -Z_tooth), (X_tip, -Z_tip_flat), (X_tip, Z_tip_flat), 
                   (X_chamfer, Z_tooth), (X_relief_top, Z_tooth), (X_relief_bot, Z_web), 
                   (self.hub_rad_m + Rf, Z_web)]
            cx, cy = self.hub_rad_m + Rf, Z_web + Rf
            mx = cx + Rf * math.cos(math.radians(225))
            my = cy + Rf * math.sin(math.radians(225))
            wp = wp.polyline(pts).threePointArc((mx, my), (self.hub_rad_m, Z_web + Rf))
            wp = wp.lineTo(self.hub_rad_m, Hub_L_m - Z_web).lineTo(Bore_m/2.0, Hub_L_m - Z_web).close()
            blank = wp.revolve(360, (0,0,0), (0,1,0))
            
        elif self.sprocket_type == 'C':
            hp = Hub_L_m / 2.0
            pts1 = [(Bore_m/2.0, -hp), (self.hub_rad_m, -hp), (self.hub_rad_m, -Z_web - Rf)]
            wp = wp.polyline(pts1)
            cx1, cy1 = self.hub_rad_m + Rf, -Z_web - Rf
            mx1 = cx1 + Rf * math.cos(math.radians(135))
            my1 = cy1 + Rf * math.sin(math.radians(135))
            wp = wp.threePointArc((mx1, my1), (self.hub_rad_m + Rf, -Z_web))
            
            wp = wp.lineTo(X_relief_bot, -Z_web).lineTo(X_relief_top, -Z_tooth).lineTo(X_chamfer, -Z_tooth)
            wp = wp.lineTo(X_tip, -Z_tip_flat).lineTo(X_tip, Z_tip_flat).lineTo(X_chamfer, Z_tooth)
            wp = wp.lineTo(X_relief_top, Z_tooth).lineTo(X_relief_bot, Z_web).lineTo(self.hub_rad_m + Rf, Z_web)
            
            cx2, cy2 = self.hub_rad_m + Rf, Z_web + Rf
            mx2 = cx2 + Rf * math.cos(math.radians(225))
            my2 = cy2 + Rf * math.sin(math.radians(225))
            wp = wp.threePointArc((mx2, my2), (self.hub_rad_m, Z_web + Rf))
            wp = wp.lineTo(self.hub_rad_m, hp).lineTo(Bore_m/2.0, hp).close()
            blank = wp.revolve(360, (0,0,0), (0,1,0))
            
        if self.sprocket_type == 'D' and self.num_bolts > 0:
            bc = self.bolt_pcd_m 
            hd = self.bolt_dia_m 
            blank = blank.cut(
                cq.Workplane("XY")
                .polarArray(bc / 2.0, 0, 360, self.num_bolts)
                .circle(hd / 2.0)
                .extrude(t1 + 0.01) 
                .translate((0, 0, -t1/2.0 - 0.005))
            )
            
        Clearance_Y = pointed_radius + 0.015 
        gap_wire = (
            cq.Workplane("XY").moveTo(P_bottom[0], P_bottom[1])
            .threePointArc(P_mid_seat, P_x).threePointArc(P_mid_work, P_y).lineTo(P_z[0], P_z[1])
            .threePointArc(P_mid_top, P_tip).lineTo(P_overcut[0], P_overcut[1])
            .lineTo(Clearance_Y, Clearance_Y).lineTo(-Clearance_Y, Clearance_Y)
            .lineTo(-P_overcut[0], P_overcut[1]).lineTo(-P_tip[0], P_tip[1])
            .threePointArc((-P_mid_top[0], P_mid_top[1]), (-P_z[0], P_z[1]))
            .lineTo(-P_y[0], P_y[1])
            .threePointArc((-P_mid_work[0], P_mid_work[1]), (-P_x[0], P_x[1]))
            .threePointArc((-P_mid_seat[0], P_mid_seat[1]), P_bottom).close()
        )
        
        gap_solid = gap_wire.extrude(Hub_L_m + 0.004).translate((0, 0, self.total_z_min - 0.002)).val()

        sprocket = blank
        for i in range(N):
            angle = i * (360.0 / N)
            cut_instance = gap_solid.translate((0, PCD / 2.0, 0)).rotate((0,0,0), (0,0,1), angle)
            sprocket = sprocket.cut(cut_instance)
            
        temp_dir = tempfile.gettempdir()
        step_path = os.path.join(temp_dir, "temp_pravaha_sprocket.step")
        cq.exporters.export(sprocket, step_path)
        return step_path

    def _extract_boundaries(self, nodes, is_3d=True):
        """Maps physical constraints dynamically across the specific Hub Z-boundaries (or 2D plane)."""
        tree_2d = cKDTree(nodes[:, :2])
        fixed_nodes = []
        bounds_min = np.full(len(nodes), 0.01)
        
        r_nodes = np.hypot(nodes[:, 0], nodes[:, 1])
        if is_3d:
            z_nodes = nodes[:, 2]
            z_min_bound = self.total_z_min - 0.001
            z_max_bound = self.total_z_max + 0.001
        else:
            z_nodes = np.zeros(len(nodes))
            z_min_bound, z_max_bound = -1.0, 1.0
        
        # 1. FIXED NODES LOGIC
        if self.sprocket_type in ['A', 'B', 'C']:
            # Fixed at the shaft bore for continuous hubs
            bore_mask = (r_nodes <= self.bore_rad_m + 0.0005) & (z_nodes >= z_min_bound) & (z_nodes <= z_max_bound)
            fixed_nodes.extend(np.where(bore_mask)[0])
        elif self.sprocket_type == 'D' and self.num_bolts > 0:
            # Fixed at the bolt holes for lug-centric
            for i in range(self.num_bolts):
                angle = i * (2.0 * math.pi / self.num_bolts)
                bx, by = (self.bolt_pcd_m / 2.0) * math.cos(angle), (self.bolt_pcd_m / 2.0) * math.sin(angle)
                dist_to_bolt = np.hypot(nodes[:, 0] - bx, nodes[:, 1] - by)
                bolt_mask = (dist_to_bolt <= (self.bolt_dia_m / 2.0) + 0.0005) & (z_nodes >= z_min_bound) & (z_nodes <= z_max_bound)
                fixed_nodes.extend(np.where(bolt_mask)[0])
                    
        num_dofs = len(nodes) * (3 if is_3d else 2)
        global_F_BOL = np.zeros(num_dofs)
        global_F_EOL = np.zeros(num_dofs)
        tooth_angle_rad = (2.0 * math.pi) / self.N_driven
        
        misalignment_deg = float(self.config.get('kinematics', {}).get('chain_misalignment_deg', 2.0))
        skew_factor = min(1.0, misalignment_deg / 2.0)
        
        for i in range(self.N_engaged):
            theta = i * tooth_angle_rad
            rx = self.R_pitch * math.cos(theta)
            ry = self.R_pitch * math.sin(theta)
            
            F_vec = self.forces_BOL_2D[i]
            F_mag = math.hypot(F_vec[0], F_vec[1])
            
            if F_mag < 1e-6: continue
                
            ux, uy = F_vec[0] / F_mag, F_vec[1] / F_mag
            contact_x = rx + (ux * self.roller_rad_m * 0.95)
            contact_y = ry + (uy * self.roller_rad_m * 0.95)
            
            flank_nodes = tree_2d.query_ball_point([contact_x, contact_y], r=0.0015)
            if not flank_nodes:
                flank_nodes = tree_2d.query_ball_point([contact_x, contact_y], r=0.0025)
                           
            if len(flank_nodes) > 0:
                mult = 3 if is_3d else 2
                f_bol_target = self.forces_BOL_3D[i] if is_3d else self.forces_BOL_2D[i]
                f_eol_target = self.forces_EOL_3D[i] if is_3d else self.forces_EOL_2D[i]
                
                if is_3d:
                    z_vals = nodes[flank_nodes, 2]
                    normalized_z = z_vals / (self.thickness_m / 2.0)
                    weights = 1.0 + (skew_factor * normalized_z)
                    weights = np.maximum(0.0, weights) 
                    weight_sum = np.sum(weights)
                    if weight_sum > 0:
                        weights = weights / weight_sum
                    else:
                        weights = np.ones(len(flank_nodes)) / len(flank_nodes)
                else:
                    weights = np.ones(len(flank_nodes)) / len(flank_nodes)
                
                for idx, n_idx in enumerate(flank_nodes):
                    patch_vec_bol = f_bol_target * weights[idx]
                    patch_vec_eol = f_eol_target * weights[idx]
                    
                    for j in range(mult):
                        global_F_BOL[n_idx*mult + j] += patch_vec_bol[j]
                        global_F_EOL[n_idx*mult + j] += patch_vec_eol[j]
                
        return list(set(fixed_nodes)), global_F_BOL, global_F_EOL

    def _apply_pillar_1_bounds(self, elements_centroids):
        bounds_min = np.full(len(elements_centroids), 0.01)
        bounds_max = np.ones(len(elements_centroids))
        r_cents = np.hypot(elements_centroids[:, 0], elements_centroids[:, 1])
        
        # 1. Lock Teeth
        R_root = self.R_pitch - self.roller_rad_m
        bounds_min[r_cents >= (R_root - 0.002)] = 1.0
        
        # --- Hub Radius Locks ---
        if self.sprocket_type in ['A', 'B', 'C']:
            # Lock entire central hub to 1.0 density
            hub_mask = r_cents <= self.hub_rad_m
            bounds_min[hub_mask] = 1.0
            
            # Absolute Void Protection for Continuous Hubs
            bore_mask = r_cents <= (self.bore_rad_m - 0.0005)
            bounds_max[bore_mask] = 0.01
            bounds_min[bore_mask] = 0.01 # Prevent boundary conflict
            
        elif self.sprocket_type == 'D' and self.num_bolts > 0:
            for i in range(self.num_bolts):
                angle = i * (2.0 * math.pi / self.num_bolts)
                bx, by = (self.bolt_pcd_m / 2.0) * math.cos(angle), (self.bolt_pcd_m / 2.0) * math.sin(angle)
                dist = np.hypot(elements_centroids[:, 0] - bx, elements_centroids[:, 1] - by)
                
                # --- THE FIX: Prevent Spoke Severing ---
                # Since the 2D mesh proxy is a solid plate, we MUST mathematically force the 
                # bolt holes to be voids, or the AI will route load-bearing spokes right through them.
                
                # 1. Solid Boss Ring (Metal)
                boss_mask = (dist <= self.boss_rad_m) & (dist > (self.bolt_dia_m / 2.0))
                bounds_min[boss_mask] = 1.0
                
                # 2. Physical Bolt Hole (Void)
                hole_mask = dist <= (self.bolt_dia_m / 2.0)
                bounds_max[hole_mask] = 0.01
                bounds_min[hole_mask] = 0.01 # Clear min bound so it overrides the default 0.01 safety
                
            # Absolute Void Protection for central axle bore
            bore_mask = r_cents <= (self.bore_rad_m - 0.0005)
            bounds_max[bore_mask] = 0.01
            bounds_min[bore_mask] = 0.01
                
        return bounds_min, bounds_max, np.sum(bounds_min == 1.0)

    def _calculate_tet4_kinematics(self, nodes, elements):
        """Vectorized 3D Volume and B-Matrix Tensor Assembly."""
        print("      -> Calculating vectorized 3D Volumes and B-Matrices...")
        elem_nodes = nodes[elements] 
        p1, p2, p3, p4 = elem_nodes[:,0,:], elem_nodes[:,1,:], elem_nodes[:,2,:], elem_nodes[:,3,:]
        
        detJ = np.einsum('ei,ei->e', p2 - p1, np.cross(p3 - p1, p4 - p1))
        volumes = np.abs(detJ) / 6.0
        
        grad_N1 = np.cross(p4-p2, p3-p2) / detJ[:, np.newaxis]
        grad_N2 = np.cross(p3-p1, p4-p1) / detJ[:, np.newaxis]
        grad_N3 = np.cross(p4-p1, p2-p1) / detJ[:, np.newaxis]
        grad_N4 = np.cross(p2-p1, p3-p1) / detJ[:, np.newaxis]
        
        B_matrices = np.zeros((len(elements), 6, 12), dtype=np.float64)
        grads = [grad_N1, grad_N2, grad_N3, grad_N4]
        
        for i in range(4):
            dx, dy, dz = grads[i][:, 0], grads[i][:, 1], grads[i][:, 2]
            c = i * 3
            B_matrices[:, 0, c] = dx
            B_matrices[:, 1, c+1] = dy
            B_matrices[:, 2, c+2] = dz
            B_matrices[:, 3, c], B_matrices[:, 3, c+1] = dy, dx
            B_matrices[:, 4, c+1], B_matrices[:, 4, c+2] = dz, dy
            B_matrices[:, 5, c], B_matrices[:, 5, c+2] = dz, dx
            
        return volumes, B_matrices
    def _setup_2d_occ_geometry(self):
        """
        Builds the 2D planar profile natively in RAM using the Gmsh GEO kernel.
        Ingests the exact procedural coordinates from the ANSI B29.1 generator.
        """
        coords = self.profile_coords[:-1] 
        num_pts = len(coords)
        
        mesh_cfg = self.config.get('continuum_meshing_3d', {})
        max_size = float(mesh_cfg.get('tet4_voxel_size_max_m', 0.005))
        min_size = float(mesh_cfg.get('tet4_voxel_size_min_m', 0.0015))
        
        if min_size > max_size:
            min_size, max_size = max_size, min_size
            
        # --- Point Sizing Sync ---
        # Apply the exact minimum element size requested in the YAML to the perimeter
        for i, (x, y) in enumerate(coords):
            gmsh.model.geo.addPoint(x, y, 0, min_size, i+1)
            
        line_tags = []
        for i in range(num_pts):
            p1 = i + 1
            p2 = (i + 1) % num_pts + 1
            tag = gmsh.model.geo.addLine(p1, p2, i+1)
            line_tags.append(tag)
            
        loop_tag = gmsh.model.geo.addCurveLoop(line_tags, 1)
        gmsh.model.geo.addPlaneSurface([loop_tag], 1)
        
        gmsh.model.geo.synchronize()
        
        # 3. Apply Boundary Layer h-refinement for the 2D mesh
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", line_tags)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 1000)
        
        # Captures the localized Root Stress correctly in the 2D ML Preconditioner
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self.mesh_size / 6.0) 
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self.mesh_size)       
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0015)               
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.006)                
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        
        gmsh.model.geo.synchronize()

    def build_2d_mesh(self):
        """Builds the 2D Planar Tri3 mesh for the rapid ML Preconditioner."""
        print("\n[Dual Mesher] Generating 2D Planar Preconditioner Mesh...")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        mesh_cfg = self.config.get('continuum_meshing_3d', {})
        max_size = float(mesh_cfg.get('tet4_voxel_size_max_m', 0.005))
        min_size = float(mesh_cfg.get('tet4_voxel_size_min_m', 0.0015))
        scale = 3.0 if self.is_coarse else 1.0
        
        if min_size > max_size:
            min_size, max_size = max_size, min_size
            
        # ---Enforce Strict Sizing ---
        # Replaces the hardcoded fallback that was ignoring the YAML
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size * scale)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size * scale)
        
        self._setup_2d_occ_geometry()
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes_2d = np.array(node_coords).reshape(-1, 3)[:, :2] 
        node_map = {tag: i for i, tag in enumerate(node_tags)}
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        raw_elements = elem_node_tags[list(elem_types).index(2)].reshape(-1, 3)
        elements_2d = np.array([[node_map[tag] for tag in elem] for elem in raw_elements], dtype=np.int32)
        
        # --- UPSTREAM 2D RCM REORDERING ---
        print("      -> Executing Reverse Cuthill-McKee (RCM) Node Reordering for VRAM Coalescing...")
        num_nodes = len(nodes_2d)
        row_indices = np.repeat(elements_2d, 3, axis=1).flatten()
        col_indices = np.tile(elements_2d, (1, 3)).flatten()
        
        # Use boolean arrays for the graph connections to minimize CPU RAM during meshing
        adj_matrix = sp.coo_matrix((np.ones(len(row_indices), dtype=np.bool_), 
                                   (row_indices, col_indices)), 
                                   shape=(num_nodes, num_nodes)).tocsr()
                                   
        perm = reverse_cuthill_mckee(adj_matrix)
        inv_perm = np.argsort(perm).astype(np.int32)
        
        # Apply permutations BEFORE boundary extraction and enforce C-contiguous memory
        nodes_2d = np.ascontiguousarray(nodes_2d[perm])
        elements_2d = np.ascontiguousarray(inv_perm[elements_2d])
        
        # Boundaries and forces are now naturally mapped to the newly RCM-sorted node order
        fixed_nodes, F_BOL, F_EOL = self._extract_boundaries(nodes_2d, is_3d=False)
        
        p1 = nodes_2d[elements_2d[:, 0]]
        p2 = nodes_2d[elements_2d[:, 1]]
        p3 = nodes_2d[elements_2d[:, 2]]
        areas = 0.5 * np.abs(p1[:,0]*(p2[:,1]-p3[:,1]) + p2[:,0]*(p3[:,1]-p1[:,1]) + p3[:,0]*(p1[:,1]-p2[:,1]))
        
        elem_centroids = (p1 + p2 + p3) / 3.0
        b_min, b_max, _ = self._apply_pillar_1_bounds(elem_centroids)
        
        gmsh.clear()
        print(f"   ✅ 2D Tri3 Mesh Complete: {len(nodes_2d)} Nodes, {len(elements_2d)} Elements.")
        
        return {
            'nodes': nodes_2d,
            'elements': elements_2d,
            'areas': areas,
            'bounds_min': b_min,
            'bounds_max': b_max,
            'fixed_nodes': fixed_nodes,
            'F_ext_BOL': F_BOL,
            'F_ext_EOL': F_EOL
        }

    def build_3d_mesh(self):
        """Builds the 3D Spatial Tet4 mesh from the CadQuery STEP file."""
        print("\n[Dual Mesher] Generating 3D Spatial Continuum Mesh via CadQuery STEP Engine...")
        
        # 1. Generate perfect STEP file
        step_file_path = self._generate_cadquery_step()
        
        # 2. Ingest into Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("Pravaha_3D_Continuum")
        
        gmsh.model.occ.importShapes(step_file_path)
        gmsh.model.occ.synchronize()
        
        # We can safely delete the temp file now that it's in RAM
        if os.path.exists(step_file_path):
            os.remove(step_file_path)
        
        mesh_cfg = self.config.get('continuum_meshing_3d', {})
        scale = 3.0 if self.is_coarse else 1.0
        
        max_size = float(mesh_cfg.get('tet4_voxel_size_max_m', 0.005)) * scale
        min_size = float(mesh_cfg.get('tet4_voxel_size_min_m', 0.0015)) * scale
        
        # --- Override OCC Native Sizing ---
        # When CAD solids are imported, the OCC kernel assigns arbitrary sizing based on curvature.
        # We must disable this and force it to respect the YAML bounds.
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        
        # Manually bind the CAD vertices to the minimum YAML size
        points = gmsh.model.getEntities(0)
        gmsh.model.mesh.setSize(points, min_size)
        
        gmsh.model.mesh.generate(3)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes_3d = np.array(node_coords).reshape(-1, 3)
        node_map = {tag: i for i, tag in enumerate(node_tags)}
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        raw_elements = elem_node_tags[list(elem_types).index(4)].reshape(-1, 4)
        elements_3d = np.array([[node_map[tag] for tag in elem] for elem in raw_elements])
        
        fixed_nodes, F_BOL, F_EOL = self._extract_boundaries(nodes_3d, is_3d=True)
        volumes, B_matrices = self._calculate_tet4_kinematics(nodes_3d, elements_3d)
        
        p1, p2, p3, p4 = nodes_3d[elements_3d[:,0]], nodes_3d[elements_3d[:,1]], nodes_3d[elements_3d[:,2]], nodes_3d[elements_3d[:,3]]
        elem_centroids = (p1 + p2 + p3 + p4) / 4.0
        b_min, b_max, count = self._apply_pillar_1_bounds(elem_centroids)
        
        print(f"   ✅ 3D Tet4 Mesh Complete: {len(nodes_3d)} Nodes, {len(elements_3d)} Elements.")
        print(f"      -> Locked {count} rigid voxels (ANSI Teeth & Hubs).")
        gmsh.clear()
        
        return {
            "nodes": nodes_3d, "elements": elements_3d, "volumes": volumes,
            "B_matrices": B_matrices, "fixed_nodes": fixed_nodes, 
            "F_ext_BOL": F_BOL, "F_ext_EOL": F_EOL,
            "bounds_min": b_min, "bounds_max": b_max
        }