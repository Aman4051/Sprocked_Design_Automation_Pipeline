import numpy as np
import math
import ezdxf
import yaml
import os
import argparse
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely import affinity
import cadquery as cq

class SprocketCAMExporter:
    """
    Transforms continuous 3D SIMP topology arrays into clean, layered, CNC-ready DXF/STEP toolpaths.
    Enforces strict ASME B29.1 Hub boundary logic based on Sprocket Type.
    """
    def __init__(self, npz_path_or_ram, config):
        print("⚙️ Initializing Material-Validated PRAVAHA CAM Exporter...")
        self.config = config
        
        if isinstance(npz_path_or_ram, str):
            data = np.load(npz_path_or_ram)
            self.densities = data['densities']
            self.nodes_mm = data['nodes'] * 1000.0
            self.elements = data['elements']
        else:
            self.densities = npz_path_or_ram['densities']
            self.nodes_mm = npz_path_or_ram['nodes'] * 1000.0
            self.elements = npz_path_or_ram['elements']
            
        geom = config.get('geometry', {})
        self.sprocket_type = str(geom.get('sprocket_type', 'C')).upper()
        
        self.bolt_pcd_mm = float(geom.get('bolt_pcd_m', 0.081)) * 1000.0
        self.num_bolts = int(geom.get('num_bolts', 4)) if self.sprocket_type == 'D' else 0
        self.bolt_dia_mm = float(geom.get('bolt_dia_m', 0.008)) * 1000.0
        self.web_thickness_mm = float(geom.get('web_thickness_m', 0.007)) * 1000.0
        
        self.num_teeth = int(geom.get('num_teeth', 39))
        self.pitch_mm = float(geom.get('pitch_inch', 0.5)) * 25.4
        self.roller_dia_mm = float(config.get('kinematics', {}).get('roller_dia_inch', 0.335)) * 25.4
        self.chain_width_mm = float(geom.get('chain_width_inch', 0.312)) * 25.4
        
        self.R_pitch_mm = self.pitch_mm / (2.0 * math.sin(math.pi / self.num_teeth))
        
        self.boss_rad_mm = self.bolt_dia_mm * 1.0 
        
        # Scale the rim margin relative to the chain pitch, ensuring it doesn't thin out on small sprockets
        self.rim_margin_mm = max(self.pitch_mm * 0.3, 6.0)
        self.bore_rad_mm = (self.bolt_pcd_mm / 2.0) - self.boss_rad_mm
        
        if geom.get('bore_dia_m') is not None:
            self.bore_rad_mm = (float(geom.get('bore_dia_m')) * 1000.0) / 2.0
            
        PD_mm = self.pitch_mm / math.sin(math.pi / self.num_teeth)
        H_m = geom.get('hub_projection_m')
        if H_m is None:
            H_inch = 0.375 + ((self.bore_rad_mm * 2.0)/25.4 / 6.0) + 0.01 * (PD_mm/25.4)
            H_mm = H_inch * 25.4
        else:
            H_mm = float(H_m) * 1000.0
            
        self.hub_rad_mm = self.bore_rad_mm + H_mm
        
        Rf_m = geom.get('hub_fillet_m')
        self.hub_fillet_mm = 0.04 * self.pitch_mm if Rf_m is None else float(Rf_m) * 1000.0
        
        Root_Dia_mm = (self.R_pitch_mm * 2.0) - self.roller_dia_mm
        if (self.hub_rad_mm * 2.0) > Root_Dia_mm - 5.0:
            self.hub_rad_mm = (Root_Dia_mm - 5.0) / 2.0
            
        self.tool_radius_mm = float(config.get('manufacturing_routing', {}).get('cnc_endmill_radius_mm', 3.175))
        self.min_web_width_mm = float(config.get('manufacturing_routing', {}).get('min_structural_web_width_m', 0.012)) * 1000.0
        
        self.bolt_coords_mm = []
        if self.sprocket_type == 'D':
            for i in range(self.num_bolts):
                angle = i * (2.0 * math.pi / self.num_bolts)
                bx = (self.bolt_pcd_mm / 2.0) * math.cos(angle)
                by = (self.bolt_pcd_mm / 2.0) * math.sin(angle)
                self.bolt_coords_mm.append((bx, by))
    
    def _determine_structural_sector_count(self, num_bolts, num_teeth):
        """
        Mirrors the Hunting Spoke / GCD logic from the Hierarchical Optimizer
        so the CAM exporter slices the exact same sectors the AI generated.
        """
        if num_bolts > 0:
            best_ns = 1
            for n in range(num_bolts, 2, -1):
                if num_bolts % n == 0:
                    best_ns = n
                    break
            return max(best_ns, 3)
        else:
            priority_spokes = [5, 7, 3, 9]
            for N_s in priority_spokes:
                if num_teeth % N_s != 0:
                    return N_s
            
            N_s = 11
            while True:
                if num_teeth % N_s != 0:
                    return N_s
                N_s += 2

    def _feature_preserving_laplacian_smooth(self, geom, iterations=60, lambda_val=0.50, protect_angle=125.0):
        """Relaxes raw voxel boundaries via curvature-adaptive Laplacian iterations."""
        if getattr(geom, 'is_empty', True): return geom
        
        enable_hub_ring = self.config.get('manufacturing_routing', {}).get('enable_hub_centering_ring', True)
        hub_margin = float(self.config.get('manufacturing_routing', {}).get('hub_centering_ring_thickness_m', 0.006)) * 1000.0
        
        locked_circles = []
        if self.sprocket_type in ['A', 'B', 'C']:
            # Lock the Hub Radius AND its Fillet
            locked_circles.append((0.0, 0.0, self.hub_rad_mm + self.hub_fillet_mm))
        else:
            locked_circles.append((0.0, 0.0, self.bore_rad_mm + (hub_margin if enable_hub_ring else 0.0)))
            for bx, by in self.bolt_coords_mm:
                locked_circles.append((bx, by, self.boss_rad_mm))

        def smooth_ring(ring):
            length = ring.length
            num_pts = int(length / 0.4) 
            if num_pts < 12: return np.array(ring.coords)
            
            pts = np.array([ring.interpolate(i/float(num_pts), normalized=True).coords[0] for i in range(num_pts)])
            weights = np.ones(num_pts)
            
            for i in range(num_pts):
                p_prev = pts[i-1]
                p_curr = pts[i]
                p_next = pts[(i+1) % num_pts]
                
                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                
                if n1 > 0 and n2 > 0:
                    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle = math.degrees(math.acos(cos_theta))
                    if angle < protect_angle:
                        weights[i] = 0.0 

            for i in range(num_pts):
                for cx, cy, cr in locked_circles:
                    dist = math.hypot(pts[i][0] - cx, pts[i][1] - cy)
                    if abs(dist - cr) < 0.30: 
                        weights[i] = 0.0
                        break

            new_pts = np.copy(pts)
            for _ in range(iterations):
                temp_pts = np.copy(new_pts)
                for i in range(num_pts):
                    if weights[i] > 0: 
                        avg_p = (temp_pts[i-1] + temp_pts[(i+1) % num_pts]) / 2.0
                        new_pts[i] = temp_pts[i] + lambda_val * weights[i] * (avg_p - temp_pts[i])
                        
            return np.vstack((new_pts, new_pts[0]))

        if geom.geom_type == 'Polygon':
            ext = smooth_ring(geom.exterior)
            ints = [smooth_ring(i) for i in geom.interiors]
            return Polygon(ext, ints).buffer(0)
            
        elif geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
            polys = [p for p in getattr(geom, 'geoms', []) if p.geom_type == 'Polygon']
            smoothed_polys = []
            for p in polys:
                ext = smooth_ring(p.exterior)
                ints = [smooth_ring(i) for i in p.interiors]
                smoothed_polys.append(Polygon(ext, ints).buffer(0))
            return unary_union(smoothed_polys)
            
        return geom

    def _generate_ai_pockets(self):
        print("   -> Isolating 3D load paths and applying strict morphological filtering...")
        
        enable_hub_ring = self.config.get('manufacturing_routing', {}).get('enable_hub_centering_ring', True)
        hub_margin_mm = float(self.config.get('manufacturing_routing', {}).get('hub_centering_ring_thickness_m', 0.006)) * 1000.0

        # Hardcode the extraction window to the known physical web thickness + a 1.0mm tolerance band
        half_web_mm = (self.web_thickness_mm / 2.0) + 1.0  
        
        strut_shapes = []
        for i, density in enumerate(self.densities):
            if density >= 0.50:  
                nodes_idx = self.elements[i]
                elem_z = np.mean(self.nodes_mm[nodes_idx, 2])
                
                # Accurately isolate elements belonging to the main web plane
                # by checking absolute distance from Z = 0.0
                if abs(elem_z) <= half_web_mm:
                    pts = self.nodes_mm[nodes_idx][:, :2]
                    if len(pts) >= 3:
                        strut_shapes.append(Polygon(pts).convex_hull)
                        
        if not strut_shapes:
            print("   [⚠️] GEOMETRY WARNING: No solid core elements detected. Reverting to solid blank.")
            return Polygon()

        raw_ai_web = unary_union(strut_shapes)
        
        collar_offset = self.min_web_width_mm / 2.0
        safe_rim_radius = self.R_pitch_mm - max(self.rim_margin_mm, collar_offset)
        
        positive_components = [raw_ai_web]
        
        # --- STRICT HUB & LUG POCKET SEPARATION ---
        if self.sprocket_type in ['A', 'B', 'C']:
            # Pockets cannot penetrate the solid shaft hub, NOR the hub fillet
            blank_inner_radius = self.hub_rad_mm + self.hub_fillet_mm + collar_offset
            positive_components.append(Point(0, 0).buffer(blank_inner_radius))
        else:
            # Type D: Pockets can go down to the bore, but must avoid the bolt bosses
            blank_inner_radius = self.bore_rad_mm
            padded_bosses = [Point(bx, by).buffer(self.boss_rad_mm + collar_offset) for bx, by in self.bolt_coords_mm]
            positive_components.append(unary_union(padded_bosses))
            if enable_hub_ring:
                safe_bore_radius = max(self.bore_rad_mm + collar_offset, self.bore_rad_mm + hub_margin_mm)
                positive_components.append(Point(0, 0).buffer(safe_bore_radius))
            # If the user disables the hub ring on a Type D sprocket, we explicitly DO NOT 
            # pad the central bore. Pockets are free to carve completely up to the void.

        # ---  VIRTUAL WELDING / STRUCTURAL FUSING ---
        # Before we carve pockets, we apply a morphological closing to the *solid* parts.
        # This acts as a virtual welding pass, flooding sharp intersections between the 
        # AI struts, the bolt bosses, and the outer rim with solid metal fillets. 
        # This guarantees the CNC endmill won't pinch off the connection points.
        fuse_radius = self.min_web_width_mm / 2.0
        raw_solid_web_padded = unary_union(positive_components).buffer(fuse_radius, join_style=1).buffer(-fuse_radius, join_style=1)
        
       # --- High-Resolution Boundary Buffers ---
        # Forces a perfectly smooth circular rim instead of a jagged default polygon
        carvable_blank = Point(0, 0).buffer(safe_rim_radius, resolution=128).difference(
            Point(0, 0).buffer(blank_inner_radius, resolution=128)
        )
        raw_pockets = carvable_blank.difference(raw_solid_web_padded)
        
        # The solid was already protected by raw_solid_web_padded. 
        # Pass the pockets directly to the symmetry generation.
        ironed_pockets = raw_pockets
        
        # Reconstruct Master Sector Symmetry based on Type
        # Query the exact same mathematical logic the optimizer used to build the topology
        symmetry_divisions = self._determine_structural_sector_count(self.num_bolts, self.num_teeth)
        master_angle = 360.0 / symmetry_divisions
        
        # Offsetting the start angle by half a sector guarantees the wedge cuts through
        # empty pocket space rather than slicing directly through the locked bolt bosses.
        offset_angle = (master_angle / 2.0) if self.sprocket_type == 'D' else 0.0
        
        wedge_radius = self.R_pitch_mm * 1.5
        wedge_pts = [(0,0)]
        for angle_deg in np.linspace(offset_angle, master_angle + offset_angle, 60): 
            rad = math.radians(angle_deg)
            wedge_pts.append((wedge_radius * math.cos(rad), wedge_radius * math.sin(rad)))
        wedge_poly = Polygon(wedge_pts)
        
        master_pockets = ironed_pockets.intersection(wedge_poly)
        
        symmetric_pockets_list = []
        for i in range(symmetry_divisions):
            rotation_angle = i * master_angle
            rotated_pocket = affinity.rotate(master_pockets, rotation_angle, origin=(0, 0))
            symmetric_pockets_list.append(rotated_pocket)
            
        perfect_symmetric_pockets = unary_union(symmetric_pockets_list).buffer(0.01, join_style=1).buffer(-0.01, join_style=1)

        # CNC Endmill Machinability Check (Morphological Opening: removes tiny voids)
        print(f"   -> Mathematically carving geometry using a {self.tool_radius_mm*2:.2f}mm Endmill...")
        tool_eroded = perfect_symmetric_pockets.buffer(-self.tool_radius_mm)
        machinable_pockets = tool_eroded.buffer(self.tool_radius_mm)

        if machinable_pockets.is_empty:
             print(f"   [⚠️] CNC LOGIC: A {self.tool_radius_mm*2:.2f}mm tool cannot fit in the current structure.")
             return Polygon() 
             
        # --- Morphological P-NUMB Filleting (Adds Metal) ---
        # Shrinks and swells the pockets to round off their outward points.
        # This explicitly ADDs solid metal fillets to the sharp inner roots of the structural web.
        multiplier = float(self.config.get('manufacturing_routing', {}).get('cnc_fillet_multiplier', 1.5))
        fillet_radius = self.tool_radius_mm * multiplier
        machinable_pockets = machinable_pockets.buffer(-fillet_radius, join_style=1).buffer(fillet_radius, join_style=1)

        print("   -> Applying continuous curvature smoothing to physical toolpaths...")
        smoothed_pockets = self._feature_preserving_laplacian_smooth(machinable_pockets, iterations=60)

        # ---  Gentle Decimation ---
        # Preserves the smooth Laplacian curves and prevents the exporter from 
        # re-introducing jagged polygonal edges based on user configuration.
        print("   -> Extracting clean boundary points for 3D extrusion...")
        export_tol = float(self.config.get('manufacturing_routing', {}).get('cnc_simplification_tolerance_mm', 0.05))
        smoothed_pockets = smoothed_pockets.simplify(tolerance=export_tol, preserve_topology=True)

        # --- HARD CONNECTIVITY GATE ---
        # Never silently ship a sprocket that pocket-carving cut into disjoint pieces.
        final_solid_check = carvable_blank.difference(smoothed_pockets)
        self._verify_structural_continuity(final_solid_check)

        return smoothed_pockets

    def _verify_structural_continuity(self, solid_2d):
        """Fails loudly instead of exporting a web that CAM has cut into disconnected islands."""
        if getattr(solid_2d, 'is_empty', True):
            raise RuntimeError(
                "CNC LOGIC FAILURE: the machined web is empty after pocket carving. "
                "The requested pockets consumed the entire structural web."
            )
        if solid_2d.geom_type == 'MultiPolygon':
            pieces = [p for p in solid_2d.geoms if p.area > 1e-6]
            if len(pieces) > 1:
                raise RuntimeError(
                    f"CNC LOGIC FAILURE: pocket carving severed the web into {len(pieces)} "
                    f"disconnected islands. A load-bearing spoke was thinner than the tool/"
                    f"web constraints allow. Increase min_structural_web_width_m, use a "
                    f"smaller cnc_endmill_radius_mm, or re-check the optimizer's eta_erode "
                    f"calibration in hierarchical_optimizer.py -- do not export this part."
                )

    def _generate_3d_cadquery_solid(self, pockets_2d):
        """Constructs the entire 3D ASME B29.1 sprocket and applies WEB-ONLY pocket cuts."""
        print(f"   -> Constructing final 3D Optimized Solid Model via CadQuery...")
        
        N = self.num_teeth
        P_mm = self.pitch_mm
        Dr_mm = self.roller_dia_mm
        W_mm = self.chain_width_mm
        Bore_mm = self.bore_rad_mm * 2.0
        
        geom = self.config.get('geometry', {})
        t1 = self.web_thickness_mm
        
        Tooth_W_mm = (0.93 * (W_mm / 25.4) - 0.006) * 25.4
        if Tooth_W_mm > t1: Tooth_W_mm = t1 
            
        Z_web = t1 / 2.0
        Z_tooth = Tooth_W_mm / 2.0
        Z_tip_flat = Z_tooth / 3.0 
        
        PCD = P_mm / math.sin(math.pi / N)
        X_tip = P_mm * (0.6 + 1.0 / math.tan(math.pi / N)) / 2.0
        X_chamfer = X_tip - (0.5 * P_mm)
        X_pitch = PCD / 2.0
        X_relief_top = X_pitch - (0.2 * P_mm)
        X_relief_bot = X_relief_top - (0.5 * P_mm)
        
        Rf = geom.get('hub_fillet_m')
        Rf = 0.04 * P_mm if Rf is None else float(Rf) * 1000.0

        if self.sprocket_type in ['B', 'C']:
            X_relief_bot = max(self.hub_rad_mm + Rf + 0.001, X_relief_bot)
        else:
            X_relief_bot = max(Bore_mm/2.0 + 0.001, X_relief_bot)
            
        H = geom.get('hub_projection_m')
        if H is None:
            H_inch = 0.375 + ((Bore_mm/25.4) / 6.0) + 0.01 * (PCD/25.4)
            H = H_inch * 25.4
        else: H = float(H) * 1000.0
            
        L = geom.get('hub_length_m')
        L = 4 * H if L is None else float(L) * 1000.0

        wp = cq.Workplane("XZ")
        
        if self.sprocket_type in ['A', 'D']:
            pts = [(self.bore_rad_mm, -Z_web), (X_relief_bot, -Z_web), (X_relief_top, -Z_tooth),
                   (X_chamfer, -Z_tooth), (X_tip, -Z_tip_flat), (X_tip, Z_tip_flat), 
                   (X_chamfer, Z_tooth), (X_relief_top, Z_tooth), (X_relief_bot, Z_web), 
                   (self.bore_rad_mm, Z_web)]
            blank = wp.polyline(pts).close().revolve(360, (0,0,0), (0,1,0))
            
        elif self.sprocket_type == 'B':
            pts = [(self.bore_rad_mm, -Z_web), (X_relief_bot, -Z_web), (X_relief_top, -Z_tooth),
                   (X_chamfer, -Z_tooth), (X_tip, -Z_tip_flat), (X_tip, Z_tip_flat), 
                   (X_chamfer, Z_tooth), (X_relief_top, Z_tooth), (X_relief_bot, Z_web), 
                   (self.hub_rad_mm + Rf, Z_web)]
            cx, cy = self.hub_rad_mm + Rf, Z_web + Rf
            mx = cx + Rf * math.cos(math.radians(225))
            my = cy + Rf * math.sin(math.radians(225))
            wp = wp.polyline(pts).threePointArc((mx, my), (self.hub_rad_mm, Z_web + Rf))
            wp = wp.lineTo(self.hub_rad_mm, L - Z_web).lineTo(self.bore_rad_mm, L - Z_web).close()
            blank = wp.revolve(360, (0,0,0), (0,1,0))
            
        elif self.sprocket_type == 'C':
            hp = L / 2.0
            pts1 = [(self.bore_rad_mm, -hp), (self.hub_rad_mm, -hp), (self.hub_rad_mm, -Z_web - Rf)]
            wp = wp.polyline(pts1)
            cx1, cy1 = self.hub_rad_mm + Rf, -Z_web - Rf
            mx1 = cx1 + Rf * math.cos(math.radians(135))
            my1 = cy1 + Rf * math.sin(math.radians(135))
            wp = wp.threePointArc((mx1, my1), (self.hub_rad_mm + Rf, -Z_web))
            
            wp = wp.lineTo(X_relief_bot, -Z_web).lineTo(X_relief_top, -Z_tooth).lineTo(X_chamfer, -Z_tooth)
            wp = wp.lineTo(X_tip, -Z_tip_flat).lineTo(X_tip, Z_tip_flat).lineTo(X_chamfer, Z_tooth)
            wp = wp.lineTo(X_relief_top, Z_tooth).lineTo(X_relief_bot, Z_web).lineTo(self.hub_rad_mm + Rf, Z_web)
            
            cx2, cy2 = self.hub_rad_mm + Rf, Z_web + Rf
            mx2 = cx2 + Rf * math.cos(math.radians(225))
            my2 = cy2 + Rf * math.sin(math.radians(225))
            wp = wp.threePointArc((mx2, my2), (self.hub_rad_mm, Z_web + Rf))
            wp = wp.lineTo(self.hub_rad_mm, hp).lineTo(self.bore_rad_mm, hp).close()
            blank = wp.revolve(360, (0,0,0), (0,1,0))

        # 2. Cut ANSI Teeth
        C_mm_const = 0.0381 
        A_rad = math.radians(35.0 + 60.0 / N)
        B_rad = math.radians(18.0 - 56.0 / N)
        M, T = 0.8 * Dr_mm * math.cos(A_rad), 0.8 * Dr_mm * math.sin(A_rad)
        W_val, V = 1.4 * Dr_mm * math.cos(math.pi / N), 1.4 * Dr_mm * math.sin(math.pi / N)
        R_val = 0.5025 * Dr_mm + C_mm_const
        E_val = 1.3025 * Dr_mm + C_mm_const
        F_val = Dr_mm * (0.8 * math.cos(B_rad) + 1.4 * math.cos(math.radians(17.0 - 64.0 / N)) - 1.3025) - C_mm_const
        
        c_cx, c_cy = -M, T
        b_cx, b_cy = W_val, -V
        
        def get_tip_int():
            sin_n = math.sin(math.pi / N)
            cos_n = math.cos(math.pi / N)
            C1 = b_cx * sin_n + (PCD/2.0 + b_cy) * cos_n
            C2 = b_cx**2 + (PCD/2.0 + b_cy)**2 - F_val**2
            t = C1 + math.sqrt(C1**2 - C2)
            return (t * sin_n, -PCD/2.0 + t * cos_n)
            
        def arc_mid(p1, p2, cx, cy, d='CCW'):
            a1 = math.atan2(p1[1] - cy, p1[0] - cx)
            a2 = math.atan2(p2[1] - cy, p2[0] - cx)
            if d == 'CCW' and a2 < a1: a2 += 2 * math.pi
            elif d == 'CW' and a2 > a1: a2 -= 2 * math.pi
            r = math.hypot(p1[0] - cx, p1[1] - cy)
            return (cx + r * math.cos((a1+a2)/2), cy + r * math.sin((a1+a2)/2))

        P_bottom = (0, -R_val)
        P_x = (R_val * math.cos(-A_rad), R_val * math.sin(-A_rad))
        cy_angle = -A_rad + B_rad
        P_y = (c_cx + E_val * math.cos(cy_angle), c_cy + E_val * math.sin(cy_angle))
        bz_angle = math.pi + cy_angle 
        P_z = (b_cx + F_val * math.cos(bz_angle), b_cy + F_val * math.sin(bz_angle))
        P_tip = get_tip_int()
        
        P_mid_seat = arc_mid(P_bottom, P_x, 0, 0, 'CCW')
        P_mid_work = arc_mid(P_x, P_y, c_cx, c_cy, 'CCW')
        P_mid_top = arc_mid(P_z, P_tip, b_cx, b_cy, 'CW')
        
        dx_tip = math.sin(math.pi / N)
        dy_tip = math.cos(math.pi / N)
        P_overcut = (P_tip[0] + 5.0 * dx_tip, P_tip[1] + 5.0 * dy_tip) 
        Clearance_Y = math.hypot(P_tip[0], P_tip[1] + PCD/2.0) + 15.0 
        
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
        
        # --- THE FIX: Use Tooth_W_mm instead of the deprecated E_mm ---
        cutter_len = max(L, Tooth_W_mm) * 2.0
        gap_solid = gap_wire.extrude(cutter_len).translate((0, 0, -cutter_len/2.0)).val()

        # --- THE INDUSTRY STANDARD: Sequential CNC Booleans ---
        # we subtract them iteratively, exactly like a real CNC machine.
        print("   -> Machining ANSI B29.1 Teeth sequentially...")
        for i in range(N):
            angle = i * (360.0 / N)
            cut_inst = gap_solid.translate((0, PCD / 2.0, 0)).rotate((0,0,0), (0,0,1), angle)
            blank = blank.cut(cut_inst)

        # 3. Punch AI Topology Pockets (Strictly through the Web)
        if not getattr(pockets_2d, 'is_empty', True):
            print("   -> Machining AI topology pockets sequentially via clean polylines...")
            pocket_polys = [pockets_2d] if pockets_2d.geom_type == 'Polygon' else [p for p in getattr(pockets_2d, 'geoms', []) if p.geom_type == 'Polygon']
            
            pocket_depth = t1 + 2.0
            pocket_z_start = -t1 / 2.0 - 1.0
            
            for poly in pocket_polys:
                ext_coords = list(poly.exterior.coords)
                # Strip the redundant closure point generated by Shapely
                if ext_coords[0] == ext_coords[-1]: ext_coords.pop() 
                
                pts = [(float(x), float(y)) for x, y in ext_coords]
                
                if len(pts) >= 3:
                    # Extrude a clean, closed polyline and subtract immediately
                    pocket_tool = cq.Workplane("XY").polyline(pts).close().extrude(pocket_depth).translate((0, 0, pocket_z_start)).val()
                    blank = blank.cut(pocket_tool)

        # 4. Drill Bolt Holes (Type D ONLY)
        if self.sprocket_type == 'D' and self.num_bolts > 0:
            print(f"   -> Drilling {self.num_bolts}x {self.bolt_dia_mm}mm bolt holes...")
            blank = blank.cut(
                cq.Workplane("XY")
                # PolarArray is already vectorized internally by CadQuery!
                .polarArray(self.bolt_pcd_mm / 2.0, 0, 360, self.num_bolts)
                .circle(self.bolt_dia_mm / 2.0)
                .extrude(t1 + 2.0)
                .translate((0, 0, -t1/2.0 - 1.0))
            )

        return blank
    
    def export_3d_step(self, pockets_2d, output_filename):
        """Generates the full 3D optimized solid and exports directly to STEP."""
        final_solid = self._generate_3d_cadquery_solid(pockets_2d)
        
        if not output_filename.lower().endswith('.step'):
            output_filename = output_filename.rsplit('.', 1)[0] + ".step"
            
        print(f"   -> Serializing solid model to STEP format: {output_filename}")
        cq.exporters.export(final_solid, output_filename)
        print("✅ Production-Ready 3D STEP Successfully Generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clean structural 3D STEP profile from PRAVAHA results.")
    
    parser.add_argument("run_name", type=str, help="The exact name of the run folder (e.g., run_20260726_143000)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_yaml = os.path.abspath(os.path.join(base_dir, "../config/sprocket_config.yaml"))
    
    # User explicitly provides the run name in the terminal
    target_dir = os.path.abspath(os.path.join(base_dir, "../results", args.run_name))
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"❌ Could not find the specified run directory: {target_dir}")

    # --- Explicitly target the Champion Trial array ---
    npz_path = os.path.join(target_dir, "topology_3d_BEST_TRIAL.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"❌ Could not find 'topology_3d_BEST_TRIAL.npz' in {target_dir}. Did the Orchestrator complete successfully?")
        
    output_path = os.path.join(target_dir, f"PRAVAHA_Production_{args.run_name}_BEST_TRIAL.step")
    
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)
        
    exporter = SprocketCAMExporter(npz_path, config)
    milling_pockets = exporter._generate_ai_pockets()
    
    # Export full 3D STEP file natively
    exporter.export_3d_step(milling_pockets, output_path)