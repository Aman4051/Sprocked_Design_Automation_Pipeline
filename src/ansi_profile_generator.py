import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union, linemerge
from shapely import affinity

class ProceduralANSIGenerator:
    """
    Generates strict ANSI standard roller chain sprocket profiles procedurally,
    using the exact internal geometry formulas from ASME B29.1.
    Passes smooth, equidistantly sampled coordinates directly to the 3D Mesher.
    """
    def __init__(self, config):
        print("   -> Initializing Strict ASME B29.1 Procedural CAD Generator...")
        self.num_teeth = int(config.get('geometry', {}).get('num_teeth', 39))
        self.pitch_m = float(config.get('geometry', {}).get('pitch_inch', 0.5)) * 0.0254
        
        # Safe-get for older config formats if kinematics dictionary is used
        if 'kinematics' in config:
            self.roller_dia_m = float(config['kinematics'].get('roller_dia_inch', 0.335)) * 0.0254
        else:
            self.roller_dia_m = float(config.get('roller_dia_inch', 0.335)) * 0.0254
            
        self.topping_curve_factor = float(config.get('manufacturing_routing', {}).get('topping_curve_factor', 0.05))
        
        # --- DYNAMIC YAML SIZING ---
        mesh_cfg = config.get('continuum_meshing_3d', {})
        self.max_size = float(mesh_cfg.get('tet4_voxel_size_max_m', 0.006))
        self.min_size = float(mesh_cfg.get('tet4_voxel_size_min_m', 0.003))
        if self.min_size > self.max_size:
            self.min_size, self.max_size = self.max_size, self.min_size
            
        # Dynamically scale Shapely polygon resolution based on the user's YAML requested element size
        OD_approx = self.pitch_m * (0.6 + 1.0 / math.tan(math.pi / self.num_teeth))
        self.res_outer = max(64, int((math.pi * OD_approx) / self.max_size))
        self.res_roller = max(16, int((math.pi * self.roller_dia_m) / self.min_size))

    def _generate_exact_tooth_gap(self):
        """Generates a single tooth gap using ASME B29.1 internal geometry."""
        P = self.pitch_m
        Dr = self.roller_dia_m
        N = self.num_teeth
        
        # Standard geometric definitions
        Ds = 1.005 * Dr + 0.003 * 0.0254 
        R_seat = Ds / 2.0
        PD = P / math.sin(math.pi / N)
        
        F = Dr * (0.8 * math.cos(math.radians(18.0 - 56.0 / N)) + 
                  1.4 * math.cos(math.radians(17.0 - 64.0 / N)) - 1.3025) - (0.0015 * 0.0254)
        H = math.sqrt(F**2 - (1.4 * Dr - P / 2.0)**2)
        
        seat_circle = Point(0, PD / 2.0).buffer(R_seat, resolution=self.res_roller)
        A_drive = math.radians(35.0 + 60.0 / N)
        A_coast = math.radians(35.0 + 60.0 / N)
        OD = P * (0.6 + 1.0 / math.tan(math.pi / N)) 
        
        extend_dist = (OD / 2.0 - PD / 2.0) + P
        Lx = -math.tan(A_coast) * extend_dist
        Rx = math.tan(A_drive) * extend_dist
        
        gap_poly = Polygon([
            (-R_seat, PD / 2.0), (R_seat, PD / 2.0),
            (Rx + R_seat, PD / 2.0 + extend_dist), (Lx - R_seat, PD / 2.0 + extend_dist)
        ])
        
        return seat_circle.union(gap_poly)

    def _generate_shapely_profile(self):
        """
        Creates the full 2D base profile of the sprocket by subtracting 
        the exactly calculated gaps from a solid OD blank.
        """
        P = self.pitch_m
        N = self.num_teeth
        
        # Approximate Outside Diameter of turned sprocket
        OD = P * (0.6 + 1.0 / math.tan(math.pi / N)) 
        
        sprocket_2d = Point(0, 0).buffer(OD / 2.0, resolution=self.res_outer)
        gaps = []
        
        base_gap = self._generate_exact_tooth_gap()
        
        for i in range(N):
            angle = i * (360.0 / N)
            gaps.append(affinity.rotate(base_gap, angle, origin=(0, 0)))
            
        # Cut the gaps out of the blank
        raw_profile = sprocket_2d.difference(unary_union(gaps))
        
        # --- MORPHOLOGICAL SMOOTHING (The Topping Curve Fix) ---
        topping_radius = P * self.topping_curve_factor 
        smooth_profile = raw_profile.buffer(-topping_radius).buffer(topping_radius)
        
        return smooth_profile

    def generate_and_export(self):
        """Extracts continuous boundary coordinates and the single gap geometry."""
        print(f"      -> Generating {self.num_teeth}T ASME B29.1 profile procedurally...")
        
        # 1. Generate the Full Profile
        profile = self._generate_shapely_profile()
        polys = [profile] if profile.geom_type == 'Polygon' else [p for p in getattr(profile, 'geoms', []) if p.geom_type == 'Polygon']
        master_poly = max(polys, key=lambda p: p.area)
        
        exterior_line = master_poly.exterior
        
        # --- COMMERCIAL CAD DECIMATION ---
        # Removes redundant collinear points from flat surfaces while preserving roots.
        # This solves the 2D mesh element explosion naturally.
        simplified_line = exterior_line.simplify(self.min_size / 4.0, preserve_topology=True)
        
        perimeter_length = simplified_line.length
        # Target spacing is now directly driven by the .yaml element sizing
        target_spacing = self.min_size * 0.8 
        num_points = max(int(perimeter_length / target_spacing), self.num_teeth * 10)
        
        coords = []
        for i in range(num_points):
            pt = simplified_line.interpolate(i / num_points, normalized=True)
            coords.append((pt.x, pt.y))
        coords.append(coords[0])
        
        # 2. Extract the Single Gap for the 3D Cutter
        base_gap = self._generate_exact_tooth_gap()
        # Smooth the gap to match the topping curve applied to the whole profile
        topping_radius = self.pitch_m * self.topping_curve_factor
        smooth_gap = base_gap.buffer(-topping_radius).buffer(topping_radius)
        
        gap_exterior = smooth_gap.exterior
        gap_length = gap_exterior.length
        gap_pts = max(int(gap_length / target_spacing), 50)
        
        gap_coords = []
        for i in range(gap_pts):
            pt = gap_exterior.interpolate(i / gap_pts, normalized=True)
            gap_coords.append((pt.x, pt.y))
        gap_coords.append(gap_coords[0])
        
        print(f"      -> Successfully extracted boundary points and 3D Cutter geometry.")
        return coords, gap_coords