#!/usr/bin/env python3
"""
Generate Cu (111) slab + porous graphene model with 8.22 nm gap
Cu at bottom, graphene above, vacuum only on top
Proper gap region definition for GCMC
"""

import numpy as np
import math
import os
from ase import Atoms
from ase.build import fcc111, add_vacuum
from scipy.spatial import KDTree
import warnings

warnings.filterwarnings("ignore")


# ========== PARAMETERS ==========
class SlabParameters:
    """Parameters for slab model generation"""

    def __init__(self):
        # Gap size (from experimental PDF)
        self.gap_size = 82.2  # Å (8.22 nm)

        # Cu parameters
        self.Cu_lattice = 3.615  # Å
        self.Cu_layers = 6  # Number of Cu layers

        # Carbon parameters
        self.carbon_layers = 3  # Number of graphene layers
        self.graphene_layer_spacing = 3.35  # Å

        # Pore parameters (from PDF)
        self.pore_diameter = 12.5  # Å (1.25 nm)
        self.pore_radius = self.pore_diameter / 2
        self.porosity = 0.3  # 30% porosity

        # Box dimensions
        self.box_x = 80.0  # Å (8 nm) - large enough for multiple pores
        self.box_y = 80.0  # Å (8 nm)

        # Vacuum only on top (above graphene)
        self.vacuum_top = 90.0  # Å vacuum above graphene

        # Output
        self.output_dir = "Cu_gap_8.22nm_model_correct"


# ========== CREATE CU (111) SLAB AT BOTTOM ==========
def create_cu_111_slab(params):
    """Create Cu (111) slab with cubic orientation using ASE"""

    print("Creating Cu (111) slab with proper FCC structure...")

    # Create Cu (111) surface with proper FCC structure
    # Size is determined by the number of unit cells needed to fill our box
    # For Cu (111), the surface lattice vectors are:
    # a1 = a/2 * [1, -1, 0]  (length = a/√2)
    # a2 = a/2 * [1, 1, -2]   (length = a√(3/2))

    a = params.Cu_lattice

    # Calculate the dimensions of the (111) surface unit cell
    # Length of a1 vector
    a1_length = a * math.sqrt(2) / 2  # a/√2
    # Length of a2 vector
    a2_length = a * math.sqrt(6) / 2  # a√(3/2)

    # Number of unit cells needed in each direction
    n_x = int(np.ceil(params.box_x / a1_length))
    n_y = int(np.ceil(params.box_y / a2_length))

    print(f"  Using {n_x}×{n_y} unit cells of Cu(111) surface")

    # Create the Cu (111) slab using ASE's built-in function
    # This creates a slab with proper FCC (111) orientation
    try:
        # Try using ASE's fcc111 function
        from ase.build import fcc111

        slab = fcc111(
            "Cu", size=(n_x, n_y, params.Cu_layers), a=a, vacuum=0, periodic=True
        )

        # Get the cell and positions
        cell = slab.get_cell()
        positions = slab.get_positions()

        print(f"  Original slab cell: {cell}")

        # Reorient the slab so the (111) surface is parallel to xy-plane
        # The fcc111 function already does this correctly

        # Scale the slab to match our desired box dimensions
        # Calculate current dimensions
        current_x = cell[0, 0]
        current_y = cell[1, 1]

        print(f"  Current dimensions: {current_x:.2f} × {current_y:.2f} Å")
        print(f"  Target dimensions: {params.box_x:.2f} × {params.box_y:.2f} Å")

        # Scale positions to match target box
        if current_x > 0 and current_y > 0:
            scale_x = params.box_x / current_x
            scale_y = params.box_y / current_y

            positions[:, 0] = positions[:, 0] * scale_x
            positions[:, 1] = positions[:, 1] * scale_y

            # Update cell
            new_cell = cell.copy()
            new_cell[0, 0] = params.box_x
            new_cell[1, 1] = params.box_y
            slab.set_cell(new_cell)
            slab.set_positions(positions)

            # Center the slab at z=0
            min_z = np.min(positions[:, 2])
            positions[:, 2] -= min_z
            slab.set_positions(positions)

        print(f"  Created Cu(111) slab with {len(slab)} atoms")
        return slab

    except ImportError:
        # Fallback: Create manual FCC (111) slab
        print("  ASE fcc111 not available, creating manual FCC (111) slab...")
        return create_manual_cu_111_slab(params)


def create_manual_cu_111_slab(params):
    """Manual creation of Cu (111) slab with FCC structure"""

    a = params.Cu_lattice

    # FCC lattice basis
    fcc_basis = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    )

    # For (111) surface, we need to rotate the lattice
    # Transformation matrix for (111) surface
    # These vectors are orthogonal for cubic simulation box

    # Vector 1: [1, -1, 0] direction (in-plane)
    v1 = np.array([1, -1, 0]) * a / math.sqrt(2)
    # Vector 2: [1, 1, -2] direction (in-plane, orthogonal to v1)
    v2 = np.array([1, 1, -2]) * a / math.sqrt(6)
    # Vector 3: [1, 1, 1] direction (out-of-plane, stack direction)
    v3 = np.array([1, 1, 1]) * a / math.sqrt(3)

    # Number of unit cells
    n1 = int(np.ceil(params.box_x / np.linalg.norm(v1)))
    n2 = int(np.ceil(params.box_y / np.linalg.norm(v2)))

    print(f"  Using {n1}×{n2} unit cells in plane")

    positions = []

    # Generate positions
    for i in range(n1):
        for j in range(n2):
            for k in range(params.Cu_layers):
                # Unit cell origin
                origin = i * v1 + j * v2 + k * v3

                # Add FCC basis atoms
                for basis in fcc_basis:
                    # Transform basis to (111) coordinates
                    # For simplicity, we'll use direct coordinates
                    pos = origin + basis[0] * v1 + basis[1] * v2 + basis[2] * v3

                    # Check if within box
                    if 0 <= pos[0] < params.box_x and 0 <= pos[1] < params.box_y:
                        positions.append(pos)

    positions = np.array(positions)

    # Remove duplicates
    if len(positions) > 0:
        # Use KDTree for fast duplicate removal
        tree = KDTree(positions[:, :2])  # Only check x,y for same layer
        mask = np.ones(len(positions), dtype=bool)

        for i in range(len(positions)):
            if mask[i]:
                # Find neighbors in xy-plane
                neighbors = tree.query_ball_point(positions[i, :2], 0.5)
                for j in neighbors:
                    if (
                        j > i
                        and mask[j]
                        and abs(positions[j, 2] - positions[i, 2]) < 0.5
                    ):
                        # Check if they're the same atom
                        dist_xy = np.linalg.norm(positions[j, :2] - positions[i, :2])
                        if dist_xy < 0.5:
                            mask[j] = False

        positions = positions[mask]

    # Create atoms object
    atoms = Atoms("Cu" * len(positions), positions=positions)

    # Set cell
    atoms.set_cell([params.box_x, params.box_y, np.max(positions[:, 2]) + 5.0])
    atoms.set_pbc([True, True, False])

    print(f"  Created manual Cu(111) slab with {len(atoms)} atoms")
    print(
        f"  Z-range: {np.min(positions[:, 2]):.1f} to {np.max(positions[:, 2]):.1f} Å"
    )

    return atoms


# ========== CREATE GRAPHENE ABOVE CU ==========
def create_graphene_above_cu(cu_atoms, params):
    """Create graphene layer above Cu with specified gap"""

    print("\nCreating graphene layer above Cu...")

    # Get Cu top surface position
    cu_positions = cu_atoms.get_positions()
    cu_z_max = np.max(cu_positions[:, 2])

    # Graphene starts at: Cu_top + gap
    graphene_z_base = cu_z_max + params.gap_size

    print(f"  Cu top surface: {cu_z_max:.1f} Å")
    print(f"  Graphene base z: {graphene_z_base:.1f} Å")
    print(f"  Gap: {params.gap_size} Å")

    # Create hexagonal graphene lattice
    a_graphene = 2.46  # Å, graphene lattice constant
    sqrt3 = np.sqrt(3)

    # Primitive vectors
    a1 = a_graphene * np.array([sqrt3, 0, 0])
    a2 = a_graphene * np.array([sqrt3 / 2, 3 / 2, 0])

    # Calculate number of unit cells needed
    n_unit_x = int(np.ceil(params.box_x / np.linalg.norm(a1))) + 2
    n_unit_y = int(np.ceil(params.box_y / np.linalg.norm(a2))) + 2

    positions = []

    # Generate positions for all layers
    for layer in range(params.carbon_layers):
        z = graphene_z_base + layer * params.graphene_layer_spacing

        for i in range(-n_unit_x, n_unit_x + 1):
            for j in range(-n_unit_y, n_unit_y + 1):
                # Unit cell position
                base = i * a1 + j * a2

                # Two atoms per graphene unit cell
                pos_a = base
                pos_b = base + a_graphene * np.array([0, 1, 0])

                positions.append([pos_a[0], pos_a[1], z])
                positions.append([pos_b[0], pos_b[1], z])

    positions = np.array(positions)

    # Wrap to periodic box in x,y
    positions[:, 0] = np.mod(positions[:, 0], params.box_x)
    positions[:, 1] = np.mod(positions[:, 1], params.box_y)

    # Remove duplicates (within 0.1 Å tolerance)
    if len(positions) > 0:
        # Use KDTree for fast duplicate removal in 2D (x,y)
        tree = KDTree(positions[:, :2])
        mask = np.ones(len(positions), dtype=bool)

        for i in range(len(positions)):
            if mask[i]:
                # Find neighbors that are too close (considering same z)
                neighbors = tree.query_ball_point(positions[i, :2], 0.1)
                for j in neighbors:
                    if (
                        j > i
                        and mask[j]
                        and abs(positions[j, 2] - positions[i, 2]) < 0.1
                    ):
                        mask[j] = False

        positions = positions[mask]

    print(f"  Base graphene atoms: {len(positions):,}")

    return positions, cu_z_max, graphene_z_base


# ========== CREATE PORES IN GRAPHENE ==========
def create_pores_in_graphene(positions, params):
    """Create pores in graphene layer"""

    if len(positions) == 0:
        return positions

    print("\nCreating pores in graphene...")

    # Calculate target number of pores
    pore_area = np.pi * params.pore_radius**2
    total_area = params.box_x * params.box_y
    n_pores_target = int(params.porosity * total_area / pore_area)

    # Ensure minimum number of pores
    n_pores = max(n_pores_target, 9)  # At least 3x3 grid

    print(f"  Target pores: {n_pores}")

    # Create pore centers on a grid
    pore_centers = []

    # Calculate grid spacing
    grid_x = int(np.sqrt(n_pores * params.box_x / params.box_y))
    grid_y = int(np.ceil(n_pores / grid_x))

    spacing_x = params.box_x / (grid_x + 1)
    spacing_y = params.box_y / (grid_y + 1)

    for i in range(1, grid_x + 1):
        for j in range(1, grid_y + 1):
            if len(pore_centers) >= n_pores:
                break

            # Add small random displacement
            dx = np.random.uniform(-spacing_x * 0.1, spacing_x * 0.1)
            dy = np.random.uniform(-spacing_y * 0.1, spacing_y * 0.1)

            pore_centers.append([i * spacing_x + dx, j * spacing_y + dy])

    # Remove atoms within pores (considering PBC)
    mask = np.ones(len(positions), dtype=bool)

    for center in pore_centers:
        # Calculate distances with minimum image convention
        dx = positions[:, 0] - center[0]
        dy = positions[:, 1] - center[1]

        # Apply PBC
        dx = dx - params.box_x * np.round(dx / params.box_x)
        dy = dy - params.box_y * np.round(dy / params.box_y)

        distances = np.sqrt(dx**2 + dy**2)
        in_pore = distances <= params.pore_radius

        mask[in_pore] = False

    porous_positions = positions[mask]

    atoms_removed = len(positions) - len(porous_positions)
    actual_porosity = atoms_removed / len(positions)

    print(f"  Removed {atoms_removed} atoms")
    print(f"  Actual porosity: {actual_porosity:.2%}")
    print(f"  Remaining atoms: {len(porous_positions):,}")

    return porous_positions


# ========== CREATE COMPLETE MODEL ==========
def create_complete_model(params):
    """Create complete model: Cu at bottom, graphene above, vacuum on top"""

    print(f"\n{'=' * 60}")
    print("CREATING MODEL: Cu (111) AT BOTTOM, GRAPHENE ABOVE, VACUUM ON TOP")
    print(f"{'=' * 60}")

    # 1. Create Cu (111) slab at z=0
    cu_atoms = create_cu_111_slab(params)
    cu_positions = cu_atoms.get_positions()
    cu_z_max = np.max(cu_positions[:, 2])

    print(f"\nCu (111) slab created:")
    print(f"  Atoms: {len(cu_atoms):,}")
    print(f"  Z-range: {np.min(cu_positions[:, 2]):.1f} to {cu_z_max:.1f} Å")

    # Calculate layer spacing for verification
    if len(cu_positions) > 0:
        # Sort by z and find distinct layers
        unique_z = np.unique(np.round(cu_positions[:, 2], 2))
        if len(unique_z) > 1:
            layer_spacing = np.mean(np.diff(np.sort(unique_z)))
            print(f"  Average layer spacing: {layer_spacing:.3f} Å")
            print(
                f"  Theoretical (111) layer spacing: {params.Cu_lattice / math.sqrt(3):.3f} Å"
            )

    # 2. Create graphene above Cu
    graphene_positions, cu_z_max_calc, graphene_z_base = create_graphene_above_cu(
        cu_atoms, params
    )

    # Verify gap calculation
    actual_gap = graphene_z_base - cu_z_max
    print(
        f"\n  Actual gap between Cu and graphene: {actual_gap:.1f} Å (target: {params.gap_size} Å)"
    )

    # 3. Create pores in graphene
    porous_graphene_positions = create_pores_in_graphene(graphene_positions, params)

    # Create graphene atoms object
    graphene_atoms = Atoms(
        "C" * len(porous_graphene_positions), positions=porous_graphene_positions
    )

    # 4. Combine Cu and graphene
    combined = cu_atoms.copy() + graphene_atoms

    # 5. Set cell with vacuum only on top
    # Box z-height: from 0 to graphene_top + vacuum
    graphene_z_max = np.max(porous_graphene_positions[:, 2])
    box_z = graphene_z_max + params.vacuum_top

    combined.set_cell([params.box_x, params.box_y, box_z])
    combined.set_pbc([True, True, False])  # Periodic in xy, fixed in z

    # Ensure atoms are within box
    final_positions = combined.get_positions()
    final_positions[:, 0] = np.mod(final_positions[:, 0], params.box_x)
    final_positions[:, 1] = np.mod(final_positions[:, 1], params.box_y)
    combined.set_positions(final_positions)

    # 6. Calculate gap region for GCMC
    # Gap region: from Cu surface to graphene bottom
    # We'll leave 1 Å buffer on each side
    gap_bottom = cu_z_max + 1.0  # 1 Å above Cu surface
    gap_top = graphene_z_base - 1.0  # 1 Å below graphene

    gap_height = gap_top - gap_bottom
    gap_volume = gap_height * params.box_x * params.box_y

    print(f"\nMODEL SUMMARY:")
    print(f"  Total atoms: {len(combined):,}")
    print(f"  Cu atoms: {len(cu_atoms):,}")
    print(f"  C atoms: {len(graphene_atoms):,}")
    print(f"  Box: {params.box_x:.1f} × {params.box_y:.1f} × {box_z:.1f} Å")
    print(f"  Cu z-range: {np.min(cu_positions[:, 2]):.1f} to {cu_z_max:.1f} Å")
    print(
        f"  Graphene z-range: {np.min(porous_graphene_positions[:, 2]):.1f} to {graphene_z_max:.1f} Å"
    )

    print(f"\nGCMC INSERTION REGION:")
    print(f"  Z-range: {gap_bottom:.1f} to {gap_top:.1f} Å")
    print(f"  Height: {gap_height:.1f} Å")
    print(f"  Volume: {gap_volume:,.0f} Å³ = {gap_volume / 1000:.2f} nm³")

    return combined, gap_bottom, gap_top, box_z


# ========== WRITE LAMMPS FILES ==========
def write_lammps_files(atoms, gap_bottom, gap_top, box_z, params):
    """Write all necessary LAMMPS files"""

    os.makedirs(params.output_dir, exist_ok=True)

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Write LAMMPS data file
    filename = f"{params.output_dir}/Cu_slab_gap_model.lammps"

    with open(filename, "w") as f:
        # Header
        f.write("LAMMPS data file: Cu (111) slab + porous graphene\n")
        f.write(
            "# Cu (111) at bottom, graphene above with 8.22 nm gap, vacuum on top\n"
        )
        f.write(f"# Box: {params.box_x:.1f} × {params.box_y:.1f} × {box_z:.1f} Å\n")
        f.write(
            f"# Gap: {gap_top - gap_bottom:.1f} Å ({gap_bottom:.1f} to {gap_top:.1f} Å)\n\n"
        )

        f.write(f"{len(atoms)} atoms\n")
        f.write(f"2 atom types\n")
        f.write(f"0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n")

        # Box
        f.write(f"0.0 {params.box_x} xlo xhi\n")
        f.write(f"0.0 {params.box_y} ylo yhi\n")
        f.write(f"0.0 {box_z} zlo zhi\n")
        f.write(f"0.0 0.0 0.0 xy xz yz\n\n")

        # Masses
        f.write("Masses\n\n")
        f.write(f"1 63.546     # Cu\n")
        f.write(f"2 12.011     # C\n\n")

        # Atoms
        f.write("Atoms # full\n\n")

        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            atom_id = i + 1
            mol_id = 1 if symbol == "Cu" else 2
            atom_type = 1 if symbol == "Cu" else 2
            charge = 0.0

            f.write(
                f"{atom_id} {mol_id} {atom_type} {charge:.6f} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n"
            )

    print(f"✓ LAMMPS data file: {filename}")

    # Write LAMMPS input script with CORRECT gap definition
    write_input_script(gap_bottom, gap_top, box_z, params)

    # Write XYZ for visualization
    from ase.io import write

    write(f"{params.output_dir}/model.xyz", atoms)

    return filename


def write_input_script(gap_bottom, gap_top, box_z, params):
    """Write LAMMPS input script with correct gap region definition"""

    # Important: In LAMMPS, region block uses INF for infinite in periodic directions
    # Since we have periodic boundaries in x and y, we use INF for those

    script = f"""# ========================================================
# LAMMPS Input for Cu (111)-Graphene Slab with 8.22 nm Gap
# Cu (111) at bottom, graphene above, vacuum on top
# Correct gap region definition for GCMC
# ========================================================

# ========== Initialization ==========
units           real
atom_style      full
boundary        p p f         # Periodic in x,y, fixed in z
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# ========== Read Structure ==========
read_data       Cu_slab_gap_model.lammps

# ========== Atom Types ==========
# Type 1: Cu (slab)
# Type 2: C  (graphene)
# Type 3: C  (CO2 molecule)
# Type 4: O  (CO2 molecule)

# Define CO2 molecule template
molecule        co2mol CO2.mol

# Set masses
mass            1 63.546     # Cu
mass            2 12.011     # C (graphene)
mass            3 12.011     # C (CO2)
mass            4 15.999     # O (CO2)

# ========== Force Field ==========
# Hybrid force field: ReaxFF for C-C, LJ for Cu-CO2
pair_style      hybrid/overlay reaxff NULL lj/cut 12.0
pair_modify     mix arithmetic

# ReaxFF for framework (adjust path to your ReaxFF file)
pair_coeff      * * reaxff ffield.reaxff Cu C NULL NULL

# IMPORTANT: Disable ReaxFF for Cu-CO2 interactions
pair_coeff      1 3 reaxff NULL
pair_coeff      1 4 reaxff NULL

# LJ for Cu-CO2 (physisorption parameters from literature)
pair_coeff      1 3 lj/cut 0.042 3.15  # Cu-C(CO2)
pair_coeff      1 4 lj/cut 0.031 3.28  # Cu-O(CO2)

# LJ for CO2-CO2 (TraPPE force field)
pair_coeff      3 3 lj/cut 0.0559 3.74   # C(CO2)-C(CO2)
pair_coeff      4 4 lj/cut 0.159 3.05    # O(CO2)-O(CO2)
pair_coeff      3 4 lj/cut 0.094 3.395   # C(CO2)-O(CO2)

# ========== Fix Bottom Cu Atoms ==========
# Fix the bottom 2 layers of Cu to prevent drift
region          cu_bottom block 0 {params.box_x} 0 {params.box_y} 0 5.0
group           cu_fixed region cu_bottom
fix             fix_cu cu_fixed setforce 0.0 0.0 0.0

# ========== GCMC Setup ==========
# CRITICAL: Define gap region correctly for GCMC
# Gap is between Cu surface and graphene
region          gcmc_region block 0 {params.box_x} 0 {params.box_y} {gap_bottom} {gap_top}

# GCMC parameters
variable        T equal 298.0      # K
variable        P equal 10.0       # bar
variable        mu equal -6.5 + 0.001987*${{T}}*ln(${{P}})  # Chemical potential

# Group for CO2 molecules
group           CO2 type 3 4

# GCMC fix - insert/delete CO2 only in the gap region
fix             gcmc_fix CO2 gcmc 100 1000 1000 0 12345 ${{T}} ${{mu}} 1.5 \\
                mol co2mol \\
                region gcmc_region \\
                pressure ${{P}} \\
                group CO2 \\
                full_energy

# ========== Thermostat ==========
group           mobile subtract all cu_fixed
fix             nvt mobile nvt temp ${{T}} ${{T}} 100.0

# ========== Monitoring ==========
# Compute number of CO2 molecules in different regions

# 1. CO2 in the entire gap
compute         co2_gap all count/region gcmc_region type 3
variable        nCO2_gap equal c_co2_gap/3  # Each CO2 has 3 atoms

# 2. CO2 near Cu surface (within 3.5 Å of Cu top)
region          cu_surface_region block 0 {params.box_x} 0 {params.box_y} {gap_bottom} {gap_bottom + 3.5}
compute         co2_cu_surface all count/region cu_surface_region type 3
variable        nCO2_cu equal c_co2_cu_surface/3

# 3. CO2 near graphene (within 3.5 Å of graphene bottom)
region          graphene_surface_region block 0 {params.box_x} 0 {params.box_y} {gap_top - 3.5} {gap_top}
compute         co2_graphene all count/region graphene_surface_region type 3
variable        nCO2_graphene equal c_co2_graphene/3

# Calculate gap density
variable        gap_volume equal ({gap_top} - {gap_bottom}) * lx * ly
variable        gap_density equal v_nCO2_gap * 1660.54 / v_gap_volume  # mmol/cm³

# Thermo output
thermo          1000
thermo_style    custom step temp density vol atoms v_nCO2_gap v_nCO2_cu v_gap_density

# File output for detailed monitoring
fix             monitor all ave/time 500 10 5000 \\
                v_nCO2_gap v_nCO2_cu v_nCO2_graphene v_gap_density \\
                file adsorption_monitor.dat

# Trajectory output
dump            traj all custom 5000 traj.lammpstrj id type x y z
dump_modify     traj sort id

# Restart files
restart         50000 restart.*.lammps

# ========== Run Protocol ==========
timestep        0.1  # 0.1 fs

# Initial minimization
min_style       cg
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

print "=== Equilibration Phase (200 ps) ==="
run             2000000  # 200 ps

print "=== Production GCMC Phase (500 ps) ==="
run             5000000  # 500 ps

# ========== Final Analysis ==========
print "========================================"
print "SIMULATION COMPLETE"
print "========================================"
print "System Specifications:"
print "  Box: {params.box_x:.1f} × {params.box_y:.1f} × {box_z:.1f} Å"
print "  Gap: {gap_top - gap_bottom:.1f} Å ({gap_bottom:.1f} to {gap_top:.1f} Å)"
print "  Gap Volume: ${{gap_volume:.0f}} Å³"
print ""
print "Adsorption Results:"
print "  CO2 in gap: ${{nCO2_gap}} molecules"
print "  CO2 near Cu surface: ${{nCO2_cu}} molecules"
print "  CO2 near graphene: ${{nCO2_graphene}} molecules"
print "  Gap density: ${{gap_density}} mmol/cm³"
print "========================================"
"""

    with open(f"{params.output_dir}/run_gcmc_simulation.lammps", "w") as f:
        f.write(script)

    print(f"✓ LAMMPS input script: {params.output_dir}/run_gcmc_simulation.lammps")

    # Also write a simple info file
    with open(f"{params.output_dir}/gap_info.txt", "w") as f:
        f.write(f"GAP REGION INFORMATION\n")
        f.write(f"=====================\n")
        f.write(f"Gap bottom (z): {gap_bottom:.2f} Å\n")
        f.write(f"Gap top (z):    {gap_top:.2f} Å\n")
        f.write(f"Gap height:     {gap_top - gap_bottom:.2f} Å\n")
        f.write(
            f"Box dimensions: {params.box_x:.1f} × {params.box_y:.1f} × {box_z:.1f} Å\n"
        )
        f.write(f"\nLAMMPS region command:\n")
        f.write(
            f"region gcmc_region block 0 {params.box_x} 0 {params.box_y} {gap_bottom} {gap_top}\n"
        )


# ========== VERIFY MODEL ==========
def verify_model(atoms, gap_bottom, gap_top, params):
    """Verify model geometry"""

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    print(f"\n{'=' * 60}")
    print("MODEL VERIFICATION")
    print(f"{'=' * 60}")

    # Check Cu atoms
    cu_mask = np.array(symbols) == "Cu"
    cu_positions = positions[cu_mask]

    if len(cu_positions) > 0:
        print(f"Cu (111) slab:")
        print(f"  Atoms: {len(cu_positions):,}")
        print(
            f"  Z-range: {np.min(cu_positions[:, 2]):.1f} to {np.max(cu_positions[:, 2]):.1f} Å"
        )
        print(f"  Bottom at z = {np.min(cu_positions[:, 2]):.1f} Å (should be ~0)")

        # Check if it's actually (111) by examining layer spacing
        unique_z = np.unique(np.round(cu_positions[:, 2], 2))
        if len(unique_z) > 1:
            layer_spacing = np.mean(np.diff(np.sort(unique_z)))
            theoretical = params.Cu_lattice / math.sqrt(3)
            print(f"  Average layer spacing: {layer_spacing:.3f} Å")
            print(f"  Theoretical (111) spacing: {theoretical:.3f} Å")
            print(f"  Difference: {abs(layer_spacing - theoretical):.3f} Å")

    # Check C atoms
    c_mask = np.array(symbols) == "C"
    c_positions = positions[c_mask]

    if len(c_positions) > 0:
        print(f"\nGraphene:")
        print(f"  Atoms: {len(c_positions):,}")
        print(
            f"  Z-range: {np.min(c_positions[:, 2]):.1f} to {np.max(c_positions[:, 2]):.1f} Å"
        )

    # Check gap
    if len(cu_positions) > 0 and len(c_positions) > 0:
        cu_top = np.max(cu_positions[:, 2])
        graphene_bottom = np.min(c_positions[:, 2])
        actual_gap = graphene_bottom - cu_top

        print(f"\nGap between Cu and graphene:")
        print(f"  Cu top: {cu_top:.1f} Å")
        print(f"  Graphene bottom: {graphene_bottom:.1f} Å")
        print(f"  Actual gap: {actual_gap:.1f} Å (target: {params.gap_size} Å)")

        # Check GCMC region
        print(f"\nGCMC insertion region:")
        print(f"  Defined: {gap_bottom:.1f} to {gap_top:.1f} Å")
        print(f"  Height: {gap_top - gap_bottom:.1f} Å")
        print(
            f"  Should be within: {cu_top + 1.0:.1f} to {graphene_bottom - 1.0:.1f} Å"
        )

        if gap_bottom < cu_top:
            print(
                f"  WARNING: Gap bottom ({gap_bottom:.1f} Å) is below Cu top ({cu_top:.1f} Å)"
            )
        if gap_top > graphene_bottom:
            print(
                f"  WARNING: Gap top ({gap_top:.1f} Å) is above graphene bottom ({graphene_bottom:.1f} Å)"
            )

    # Check vacuum
    all_z_max = np.max(positions[:, 2])
    box_z = atoms.get_cell()[2, 2]
    vacuum = box_z - all_z_max

    print(f"\nVacuum:")
    print(f"  Highest atom: {all_z_max:.1f} Å")
    print(f"  Box height: {box_z:.1f} Å")
    print(f"  Vacuum on top: {vacuum:.1f} Å")

    if vacuum < 10.0:
        print(f"  WARNING: Vacuum might be too small ({vacuum:.1f} Å)")

    print(f"{'=' * 60}")


# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTED SLAB MODEL - Cu (111) BOTTOM, GRAPHENE ABOVE")
    print("=" * 60)

    # Initialize parameters
    params = SlabParameters()

    print(f"\nModel specifications:")
    print(f"  Box XY: {params.box_x:.1f} × {params.box_y:.1f} Å")
    print(f"  Gap: {params.gap_size} Å (8.22 nm)")
    print(f"  Cu layers: {params.Cu_layers} (FCC (111) orientation)")
    print(f"  Carbon layers: {params.carbon_layers}")
    print(f"  Pore diameter: {params.pore_diameter} Å")
    print(f"  Porosity: {params.porosity:.1%}")
    print(f"  Vacuum on top: {params.vacuum_top} Å")

    # Create model
    atoms, gap_bottom, gap_top, box_z = create_complete_model(params)

    # Verify model
    verify_model(atoms, gap_bottom, gap_top, params)

    # Write files
    write_lammps_files(atoms, gap_bottom, gap_top, box_z, params)

    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {params.output_dir}")
    print(f"\nFiles created:")
    print(f"  1. Cu_slab_gap_model.lammps - Structure with Cu (111)")
    print(f"  2. run_gcmc_simulation.lammps - LAMMPS input with CORRECT gap definition")
    print(f"  3. gap_info.txt - Gap region information")
    print(f"  4. model.xyz - Visualization file")

    print(f"\nKey features:")
    print(f"  • Cu (111) slab at bottom (z ≈ 0)")
    print(f"  • Graphene above with {params.gap_size} Å gap")
    print(f"  • Vacuum only on top of graphene")
    print(f"  • Correct GCMC region: z = {gap_bottom:.1f} to {gap_top:.1f} Å")

    print(f"\nTo run simulation:")
    print(f"  cd {params.output_dir}")
    print(f"  mpirun -np 4 lammps -in run_gcmc_simulation.lammps")
    print("=" * 60)

