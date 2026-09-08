import os
import argparse
import numpy as np
import pyvista as pv

class PravahaVisualizer:
    """
    Decoupled VTK Visualization Engine for PRAVAHA V2.
    Converts raw mathematical .npz arrays into interactive 3D 
    unstructured grids mapping Stress, Density, and Deflection.
    """
    def __init__(self, npz_path):
        print(f"👁️ Initializing 3D VTK Visualizer for: {os.path.basename(npz_path)}")
        self.npz_path = npz_path
        self.output_vtu = npz_path.replace('.npz', '.vtu')
        
    def load_and_build_vtk(self):
        print("   -> Extracting Tensors and Constructing Unstructured Tet4 Grid...")
        # 1. Load the exported mathematical artifacts
        data = np.load(self.npz_path)
        nodes = data['nodes']
        elements = data['elements']
        densities = data['densities']
        stresses = data['stresses']
        
        # Deflections are flattened (N * 3), reshape to (N, 3) vectors
        deflections = data['deflections'].reshape(-1, 3)
        
        # 2. Format elements for VTK/PyVista
        # VTK requires unstructured cells to be prefixed by the number of points per cell.
        # Since we use Tet4 elements, we prepend '4' to every element array.
        vtk_cells = np.hstack((np.full((len(elements), 1), 4), elements)).flatten()
        cell_types = np.full(len(elements), pv.CellType.TETRA, dtype=np.uint8)
        
        # 3. Build the Mesh Object
        self.grid = pv.UnstructuredGrid(vtk_cells, cell_types, nodes)
        
        # 4. Attach Physics Tensors
        # Elements hold Density and Von Mises Stress
        self.grid.cell_data['Density'] = densities
        self.grid.cell_data['Von Mises Stress (Pa)'] = stresses
        
        # Nodes hold the spatial Deflection vectors
        self.grid.point_data['Deflection Vector (m)'] = deflections
        
        # Calculate scalar magnitude of deflection for color mapping
        self.grid.point_data['Deflection Magnitude (m)'] = np.linalg.norm(deflections, axis=1)
        
    def visualize(self):
        print("   -> Filtering out AI 'Void' space (Density < 0.50)...")
        # Apply the exact same Heaviside projection threshold used in the orchestrator
        solid_mesh = self.grid.threshold(value=0.50, scalars='Density')
        
        # Save to VTU so the user can open it in ParaView for deep academic inspection
        solid_mesh.save(self.output_vtu)
        print(f"   ✅ Saved VTK file for ParaView: {self.output_vtu}")
        
        print("   -> Launching Interactive PyVista Plotter...")
        # Configure a dual-window plotter
        plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
        
        # --- Left Window: Von Mises Stress ---
        plotter.subplot(0, 0)
        plotter.add_text("Physical Von Mises Stress (Pa)", font_size=12)
        plotter.add_mesh(solid_mesh, scalars='Von Mises Stress (Pa)', cmap='turbo', 
                         show_edges=False, scalar_bar_args={'title': 'Stress (Pa)'})
        plotter.add_axes()
        
        # --- Right Window: Deformation ---
        plotter.subplot(0, 1)
        plotter.add_text("Deflection Magnitude (m) [Warped 100x]", font_size=12)
        # Warp the mesh by its deflection vector to physically visualize the bending
        warped_mesh = solid_mesh.warp_by_vector('Deflection Vector (m)', factor=100.0)
        plotter.add_mesh(warped_mesh, scalars='Deflection Magnitude (m)', cmap='plasma', 
                         show_edges=False, scalar_bar_args={'title': 'Deflection (m)'})
        
        # Link the cameras so rotating one window rotates the other
        plotter.link_views()
        plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PRAVAHA 3D .npz artifacts.")
    parser.add_argument("--run_dir", type=str, help="Path to run directory", required=False)
    args = parser.parse_args()

    # Automatically find the latest completed run if no argument is provided
    if not args.run_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_base = os.path.abspath(os.path.join(base_dir, "../results"))
        
        candidate_dirs = [os.path.join(results_base, d) for d in os.listdir(results_base) if d.startswith("run_")]
        completed_runs = [d for d in candidate_dirs if os.path.exists(os.path.join(d, "topology_3d.npz"))]
        
        if not completed_runs:
            raise FileNotFoundError("❌ No completed run with a topology_3d.npz was found.")
            
        completed_runs.sort(key=lambda d: os.path.getmtime(os.path.join(d, "topology_3d.npz")))
        target_dir = completed_runs[-1]
        print(f"   -> Auto-selected latest COMPLETED run: {target_dir}")
    else:
        target_dir = args.run_dir
        
    npz_file = os.path.join(target_dir, "topology_3d.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"❌ Could not find {npz_file}")
        
    visualizer = PravahaVisualizer(npz_file)
    visualizer.load_and_build_vtk()
    visualizer.visualize()