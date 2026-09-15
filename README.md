GPU-Accelerated Topology Optimization Suite

[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CuPy](https://img.shields.io/badge/CuPy-Enabled-brightgreen.svg)](https://cupy.dev/)
[![Gmsh](https://img.shields.io/badge/Gmsh-OCC%20Engine-orange.svg)](https://gmsh.info/)
[![CadQuery](https://img.shields.io/badge/CadQuery-Parametric%20CAD-red.svg)](https://cadquery.readthedocs.io/)

This is an end-to-end, high-performance computational design suite engineered for the generative design and structural optimization of sprockets. Specifically tailored for **ASME B29.1 roller chain sprockets**, it couples custom GPU-accelerated matrix-free finite element kernels, automated DFM (CNC) toolpath filters, and autonomous FEA validation using Salome Code_Aster.

---

## ⚡ Core Technical Innovations

### 1. Matrix-Free In-Core GPU Solvers (Zero-Copy SpMV)
* **Custom CUDA C++ Kernels:** Bypasses CSR assembly bottlenecks by executing element-by-element (EBE) SpMV operations directly in GPU VRAM.
* **Structure of Arrays (SoA) Coalescing:** Employs custom kernels for 3D linear tetrahedral elements (Tet4, 12x12 stiffness) and 2D constant strain triangles (Tri3, 6x6 full and 21-entry sub-warp symmetric packing via warp shuffles).
* **Advanced Preconditioning:** Utilizes a 3rd-degree damped Neumann polynomial preconditioner alongside Reverse Cuthill-McKee (RCM) degree-of-freedom ordering to minimize PCG iterations across extreme SIMP stiffness contrasts.

### 2. Hierarchical Optimization
* **Phase 1 (2D Planar Sprint):** Rapidly evolves the global load-bearing blueprint on a 2D Tri3 mesh using Optimality Criteria (OC) before projecting into 3D space.
* **Phase 2 & 3 (Spatial Polish):** Projects the 2D topology to a 3D Tet4 preconditioner via KD-Trees, finalizing out-of-plane refinements directly on the 3D continuum (but this part isn't implemented properly).

### 3. Kinematic & Manufacturing-Aware Formulations
* **Rotational Inertia ($I_z$) Minimization:** Dynamically scales elemental compliance gradients with a radial mass moment of inertia penalty ($r^2$).
* **Hunting Spoke & Cyclic Symmetry:** Dynamically determines cyclically symmetric sectors ($N_s$) via Greatest Common Divisor (GCD) for bolted hubs (Type D) or acoustic hunting spoke algorithms for continuous hubs (Types A, B, C).
* **Manufacturability & Length-Scale Control:** Calibrated Heaviside projection thresholds ($\eta_{\text{dilate}}, \eta_{\text{erode}}$) prevent spoke pinching below CNC endmill minimum widths.
* **Feature-Preserving Laplacian Boundary Smoothing:** Translates discrete voxel clusters into smooth, tangent-continuous spline boundaries while preserving critical bolt boss and hub geometry.

### 4. Autonomous Validation & Digital Twin Harvester
* **Outer-Loop FSD Bisection:** Actively scales global volume fractions and tunes inertia weights ($\gamma$) against multi-objective constraints, including Web VMIS FoS, Tooth Shear FoS, in-plane deflection, and axial runout.
* **Decoupled Aster/Salome Verification:** Mid-loop geometry is automatically meshed with quadratic elements and verified in Salome/Code_Aster, subsequently triggering autonomous volume/inertia tuning in the cycle.
* **ML Data Generation:** Harvests full 3D continuous density arrays, von Mises stress tensors, and displacement fields at every trial
 for downstream surrogate modeling.

### `sprocket_config.yaml`
Acts as the single source of truth for the optimization pipeline[cite: 12]. It is divided into distinct parameter blocks:
* **Geometry & ASME B29.1 Overrides:** Defines the core sprocket type (A, B, C, or D), tooth count, pitch, bore diameter, and bolt patterns[cite: 12].
* **Drivetrain Kinematics:** Sets the operational environment, including maximum motor RPM, chain tension, slack percentage, and misalignment allowances[cite: 12].
* **Topology Constraints:** Governs the autonomous outer-loop behavior, establishing target Factors of Safety (FoS) for the structural web and gear teeth, maximum deflection limits, plateau detection rules, and the rotational inertia penalty weight ($\gamma$)[cite: 12].
* **Manufacturing & DFM:** Configures the CNC endmill radius, minimum allowable web widths, and hub centering ring settings for the post-processor[cite: 12].
* **Continuum Meshing:** Defines the maximum and minimum spatial resolution limits for the 3D voxel grid[cite: 12].

### `materials_library.yaml`
A flattened, easily extensible materials database containing isotropic mechanical properties[cite: 11].
* Stores critical values required by the FEA solver, including mass density, Poisson's ratio, Young's modulus, and yield strength[cite: 11].
* The active material is dynamically selected by the `active_material` key in `sprocket_config.yaml`[cite: 12].

## 💻 System Requirements

### Hardware
* **GPU**: NVIDIA GPU with Compute Capability 7.0 or higher (Volta, Turing, Ampere, Ada Lovelace, or Hopper architectures). The CUDA kernels utilize `__shfl_sync` warp-level primitives and `atomicAdd` for double-precision floats[cite: 1]. 
* **VRAM**: 8 GB minimum. The in-core matrix-free solver loads element-by-element tensors directly into VRAM (e.g., a standard run stores a ~330MB tensor matrix in memory alongside SpMV scratch buffers)[cite: 4]. 12 GB+ is recommended for massive continuum meshes.
* **RAM**: 16 GB+ recommended for CadQuery boundary evaluations and Gmsh 3D voxelization[cite: 3, 6].

### Software & Python Dependencies
* **OS**: Linux (Ubuntu 20.04/22.04/24.04) or Windows via WSL2.
* **CUDA Toolkit**: 11.8 or 12.x.
* **Python**: 3.10+.
* **Core Libraries**: 
  * `cupy` and `cupyx` for GPU array manipulation and JPCG linear operators.
  * `numpy` and `scipy` for RCM graph reordering and KD-Tree spatial mapping[cite: 3, 4, 6, 7].
  * `gmsh` and `cadquery` for strict ASME B29.1 procedural CAD generation and boundary layer meshing[cite: 3, 6, 9].
  * `shapely` and `ezdxf` for 2D CNC toolpath generation and Laplacian morphological smoothing[cite: 9, 10].
  * `pyyaml` for parsing configuration libraries[cite: 8, 9].

Install dependencies via pip:
```bash
pip install numpy scipy cupy-cuda12x gmsh cadquery shapely ezdxf pyyaml
```
## 🛠️ Build & Compilation

If you prefer to bypass CMake and compile the C++ GPU engine directly, you can compile with the NVIDIA CUDA Compiler (`nvcc`).

### Direct Compilation via NVCC
Navigate to the root or `src/` directory and compile `cuda_solver.cu` into a shared library. The kernel targets modern NVIDIA architectures (Ada Lovelace `sm_89` by default, but compatible with Volta `sm_70` and newer)[cite: 1, 2]:

```bash
# Compile directly into a shared object library
nvcc -O3 -shared -Xcompiler -fPIC \
     -arch=sm_89 \
     src/cuda_solver.cu -o build/pravaha_cu_engine.so
# Run the pipeline from the repository root
python src/main_orchestrator.py
# Pass the timestamped folder name from the results directory (or just run folder name)
python src/manufacturing_export.py run_YYYYMMDD_HHMMSS
```
