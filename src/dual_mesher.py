import math
import numpy as np
import gmsh
from scipy.spatial import cKDTree

class DualContinuumMesher:
    """
    PRAVAHA V2 Unified 2D/3D OpenCASCADE Mesher.
    Generates the low-fidelity 2D Tri3 mesh for the rapid spatial preconditioner, 
    and the high-fidelity 3D Tet4 mesh for the final GPU spatial polish.
    """
    def __init__(self, config, physics_payload, profile_coords):
        print("   -> Initializing Unified Dual Continuum Mesher...")
        self.config = config
        self.payload = physics_payload
        self.profile_coords = profile_coords
        
        self.bolt_pcd_m = float(self.config.get('bolt_pcd_m', 0.081))
        self.num_bolts = int(self.config.get('num_bolts', 4))
        self.thickness_m = float(self.config.get('web_thickness_m', 0.007))
        self.bolt_dia_m = float(self.config.get('bolt_dia_m', 0.008))
        self.N_driven = int(self.config.get('num_teeth', 39))
        
        self.mesh_size = float(self.config.get('continuum_meshing_3d', {}).get('starting_voxel_size_m', 0.003))

    def _setup_occ_geometry(self, extrude_3d=False):
        """Builds the 2D/3D solid natively in RAM using the robust Gmsh GEO kernel."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size * 0.5)
        
        coords = self.profile_coords[:-1] 
        num_pts = len(coords)
        
        # 1. Map hyper-dense mathematical points (Bypasses OCC spline ringing)
        for i, (x, y) in enumerate(coords):
            gmsh.model.geo.addPoint(x, y, 0, self.mesh_size, i+1)
            
        # 2. Connect into a continuous high-resolution wireframe
        line_tags = []
        for i in range(num_pts):
            p1 = i + 1
            p2 = (i + 1) % num_pts + 1
            tag = gmsh.model.geo.addLine(p1, p2, i+1)
            line_tags.append(tag)
            
        loop_tag = gmsh.model.geo.addCurveLoop(line_tags, 1)
        surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag], 1)
        
        gmsh.model.geo.synchronize()
        
        # 3. Apply the Physics Constraints (Structured Extrusion & Boundary Layer)
        if extrude_3d:
            # Prevent Tet4 Shear Locking 
            num_z_layers = max(5, int(self.thickness_m / 0.0015))
            extrude_entities = gmsh.model.geo.extrude(
                [(2, surface_tag)], 0, 0, self.thickness_m, 
                numElements=[num_z_layers]
            )
            gmsh.model.geo.synchronize()
            
            # Extract 3D Flanks (In geo.extrude, side surfaces are all dim=2 entities after the top surface)
            side_surfaces = [tag for dim, tag in extrude_entities[1:] if dim == 2]
            
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "SurfacesList", side_surfaces)
            gmsh.model.mesh.field.setNumber(1, "Sampling", 1000)
            
        else:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "CurvesList", line_tags)
            gmsh.model.mesh.field.setNumber(1, "Sampling", 1000)
        
        # Boundary Layer h-refinement (Captures the localized Root Stress)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self.mesh_size / 6.0) 
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self.mesh_size)       
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0015)               
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.006)                
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        
        if extrude_3d:
            gmsh.model.geo.synchronize()

    def _map_boundaries(self, nodes, is_3d=True):
        """
        Calculates Fixed Nodes (Bolt Holes), Solid Boss constraints, 
        and maps the discrete physics payload vectors directly to the tooth mesh.
        """
        num_dofs = len(nodes) * (3 if is_3d else 2)
        F_BOL = np.zeros(num_dofs, dtype=np.float64)
        F_EOL = np.zeros(num_dofs, dtype=np.float64)
        fixed_nodes = []
        bounds_min = []
        
        # --- NEW: Calculate the Root Radius for Non-Design Space Protection ---
        R_pitch = self.payload['R_pitch']
        R_roller = (self.config.get('roller_dia_inch', 0.335) * 0.0254) / 2.0
        R_root = R_pitch - R_roller
        
        # 1. Map Fixed Boundaries, Boss Constraints & Non-Design Tooth Domain
        for i, pt in enumerate(nodes):
            is_boss = False
            for j in range(self.num_bolts):
                angle = j * (2.0 * math.pi / self.num_bolts)
                bx = (self.bolt_pcd_m / 2.0) * math.cos(angle)
                by = (self.bolt_pcd_m / 2.0) * math.sin(angle)
                dist = math.hypot(pt[0] - bx, pt[1] - by)
                
                # If node is exactly on the bolt wall (Fixed Boundary)
                if dist <= (self.bolt_dia_m / 2.0) * 1.05:
                    fixed_nodes.append(i)
                    
                # If node is within the structural boss (Force Solid Constraint)
                if dist <= (self.bolt_dia_m / 2.0) * 2.5:
                    is_boss = True
                    
            # --- THE FIX: PROTECT THE TEETH ---
            # Any node in the tooth profile or the 2mm sub-surface root fillet 
            # is strictly forbidden from being hollowed out by the optimizer.
            r = math.hypot(pt[0], pt[1])
            is_tooth = r >= (R_root - 0.002)
            
            # Lock both the bolt bosses AND the teeth to solid metal (1.0)
            bounds_min.append(1.0 if (is_boss or is_tooth) else 0.01)

        # 2. Map Dual-Case Load Envelope Vectors to the Tooth Mesh
        # --- THE FIX: 2D CYLINDRICAL PROJECTION TREE ---
        # By querying the X,Y coordinates only, we create a continuous 
        # Z-axis cylinder, perfectly mimicking a physical chain roller.
        tree_2d = cKDTree(nodes[:, :2])
        
        alpha = (2.0 * math.pi) / self.N_driven
        N_eng = self.payload['N_engaged']
        
        f_bol_key = 'F_spatial_BOL' if is_3d else 'F_planar_BOL'
        f_eol_key = 'F_spatial_EOL' if is_3d else 'F_planar_EOL'
        
        vecs_bol = self.payload[f_bol_key]
        vecs_eol = self.payload[f_eol_key]
        
        # Calculate the physical roller radius
        R_roller = (self.config.get('roller_dia_inch', 0.335) * 0.0254) / 2.0
        R_pitch = self.payload['R_pitch']
        
        # Distribute point-loads to simulate roller contact patch
        for i in range(N_eng):
            # Theoretical center of the chain roller on the Pitch Circle
            theta = i * alpha
            Rx = R_pitch * math.cos(theta)
            Ry = R_pitch * math.sin(theta)
            
            # Extract the physical force vector
            F_vec = vecs_bol[i]
            F_mag = math.hypot(F_vec[0], F_vec[1])
            
            if F_mag < 1e-6:
                continue 
                
            # Vector Projection: Project contact point onto the driving FLANK
            dir_x = F_vec[0] / F_mag
            dir_y = F_vec[1] / F_mag
            
            pt_x = Rx + (dir_x * R_roller * 0.95)
            pt_y = Ry + (dir_y * R_roller * 0.95)
            
            if is_3d:
                # --- CONTINUOUS LINE LOAD ---
                # A 1.5mm 2D radius grabs a continuous 3mm wide cylinder across 
                # the entire 7mm Z-axis. No gaps, no punching singularities.
                nearest_nodes = tree_2d.query_ball_point([pt_x, pt_y], r=0.0015) 
                
                # Dynamic Expansion Fallback
                if not nearest_nodes:
                    nearest_nodes = tree_2d.query_ball_point([pt_x, pt_y], r=0.0025)
                    
                if not nearest_nodes:
                    _, nearest_nodes = tree_2d.query([pt_x, pt_y], k=20)
            else:
                _, nearest_nodes = tree_2d.query([pt_x, pt_y], k=4)
                
            # Divide the total force equally among the continuous patch
            patch_vec_bol = vecs_bol[i] / len(nearest_nodes)
            patch_vec_eol = vecs_eol[i] / len(nearest_nodes)
            
            for n_idx in nearest_nodes:
                if is_3d:
                    F_BOL[n_idx*3 : n_idx*3+3] += patch_vec_bol
                    F_EOL[n_idx*3 : n_idx*3+3] += patch_vec_eol
                else:
                    F_BOL[n_idx*2 : n_idx*2+2] += patch_vec_bol
                    F_EOL[n_idx*2 : n_idx*2+2] += patch_vec_eol

        return np.array(fixed_nodes, dtype=np.int32), F_BOL, F_EOL, np.array(bounds_min, dtype=np.float64)
    
    def _calculate_tet4_kinematics(self, nodes, elements):
        """
        Vectorized 3D Volume and B-Matrix Tensor Assembly.
        Calculates the 6x12 Shape Function Gradients instantly using pure Numpy cross-products,
        bypassing the slow element-by-element Python loops.
        """
        print("      -> Calculating vectorized 3D Volumes and B-Matrices...")
        elem_nodes = nodes[elements] 
        
        p1 = elem_nodes[:, 0, :]
        p2 = elem_nodes[:, 1, :]
        p3 = elem_nodes[:, 2, :]
        p4 = elem_nodes[:, 3, :]
        
        # Tet4 Volume Calculation using Determinant/Cross Products
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p4 - p1
        detJ = np.einsum('ei,ei->e', v1, np.cross(v2, v3))
        volumes = np.abs(detJ) / 6.0
        
        # Shape Function Gradients (Inverted Jacobian Cofactors)
        grad_N1 = np.cross(p4-p2, p3-p2) / detJ[:, np.newaxis]
        grad_N2 = np.cross(p3-p1, p4-p1) / detJ[:, np.newaxis]
        grad_N3 = np.cross(p4-p1, p2-p1) / detJ[:, np.newaxis]
        grad_N4 = np.cross(p2-p1, p3-p1) / detJ[:, np.newaxis]
        
        # Construct the (E, 6, 12) Symmetric Strain Tensor (B)
        B_matrices = np.zeros((len(elements), 6, 12), dtype=np.float64)
        grads = [grad_N1, grad_N2, grad_N3, grad_N4]
        
        for i in range(4):
            dx, dy, dz = grads[i][:, 0], grads[i][:, 1], grads[i][:, 2]
            c = i * 3
            B_matrices[:, 0, c] = dx
            B_matrices[:, 1, c+1] = dy
            B_matrices[:, 2, c+2] = dz
            B_matrices[:, 3, c] = dy
            B_matrices[:, 3, c+1] = dx
            B_matrices[:, 4, c+1] = dz
            B_matrices[:, 4, c+2] = dy
            B_matrices[:, 5, c] = dz
            B_matrices[:, 5, c+2] = dx
            
        return volumes, B_matrices

    def build_2d_mesh(self):
        """Builds the 2D Planar Tri3 mesh for the rapid ML Preconditioner."""
        print("\n[Dual Mesher] Generating 2D Planar Preconditioner Mesh...")
        self._setup_occ_geometry(extrude_3d=False)
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes_2d = np.array(node_coords).reshape(-1, 3)[:, :2] # Drop Z
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri3_idx = np.where(elem_types == 2)[0][0]
        elements_2d = np.array(elem_node_tags[tri3_idx]).reshape(-1, 3) - 1 # 0-indexed
        
        fixed_nodes, F_BOL, F_EOL, bounds_min = self._map_boundaries(nodes_2d, is_3d=False)
        
        # Approximate 2D element area via cross product
        p1 = nodes_2d[elements_2d[:, 0]]
        p2 = nodes_2d[elements_2d[:, 1]]
        p3 = nodes_2d[elements_2d[:, 2]]
        areas = 0.5 * np.abs(p1[:,0]*(p2[:,1]-p3[:,1]) + p2[:,0]*(p3[:,1]-p1[:,1]) + p3[:,0]*(p1[:,1]-p2[:,1]))
        
        # Elements are constrained based on their centroid's location
        elem_centroids = (p1 + p2 + p3) / 3.0
        _, _, _, elem_bounds_min = self._map_boundaries(elem_centroids, is_3d=False)
        
        gmsh.clear()
        print(f"   ✅ 2D Tri3 Mesh Complete: {len(nodes_2d)} Nodes, {len(elements_2d)} Elements.")
        
        return {
            'nodes': nodes_2d,
            'elements': elements_2d,
            'areas': areas,
            'bounds_min': elem_bounds_min,
            'bounds_max': np.ones(len(elements_2d)),
            'fixed_nodes': fixed_nodes,
            'F_ext_BOL': F_BOL,
            'F_ext_EOL': F_EOL
        }

    def build_3d_mesh(self):
        """Builds the 3D Spatial Tet4 mesh for the high-fidelity GPU C++ polish."""
        print("\n[Dual Mesher] Generating 3D Spatial Continuum Mesh...")
        self._setup_occ_geometry(extrude_3d=True)
        gmsh.model.mesh.generate(3)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes_3d = np.array(node_coords).reshape(-1, 3)
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        tet4_idx = np.where(elem_types == 4)[0][0]
        elements_3d = np.array(elem_node_tags[tet4_idx]).reshape(-1, 4) - 1
        
        fixed_nodes, F_BOL, F_EOL, _ = self._map_boundaries(nodes_3d, is_3d=True)
        
        # High-Speed Vectorized Assembly
        volumes, B_matrices = self._calculate_tet4_kinematics(nodes_3d, elements_3d)
        
        p1 = nodes_3d[elements_3d[:, 0]]
        p2 = nodes_3d[elements_3d[:, 1]]
        p3 = nodes_3d[elements_3d[:, 2]]
        p4 = nodes_3d[elements_3d[:, 3]]
        elem_centroids = (p1 + p2 + p3 + p4) / 4.0
        _, _, _, elem_bounds_min = self._map_boundaries(elem_centroids, is_3d=True)
        
        gmsh.clear()
        print(f"   ✅ 3D Tet4 Mesh Complete: {len(nodes_3d)} Nodes, {len(elements_3d)} Elements.")
        print(f"   ✅ 3D B-Matrices successfully compiled for C++ Engine.")
        
        return {
            'nodes': nodes_3d,
            'elements': elements_3d,
            'volumes': volumes,
            'B_matrices': B_matrices,
            'bounds_min': elem_bounds_min,
            'bounds_max': np.ones(len(elements_3d)),
            'fixed_nodes': fixed_nodes,
            'F_ext_BOL': F_BOL,
            'F_ext_EOL': F_EOL
        }