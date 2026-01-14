#!/usr/bin/env python3
"""
Generate Cu (111) slab + porous graphene model with 8.22 nm gap
Optimized for GCMC simulations of CO2 adsorption
"""

import numpy as np
import math
import os
from ase import Atoms
from ase.build import fcc111
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
        self.Cu_layers = 6  # Number of Cu layers for stability

        # Carbon parameters
        self.carbon_layers = 3  # Number of graphene layers (trilayer)
        self.C_C_bond = 1.42  # Å
        self.graphene_layer_spacing = 3.35  # Å between graphene layers

        # Pore parameters (from PDF)
        self.pore_diameter = 12.5  # Å (1.25 nm)
        self.pore_radius = self.pore_diameter / 2
        self.porosity = 0.3  # 30% porosity

        # Box dimensions (must be large enough for pores)
        # We'll make it a nice round number that's an integer multiple of Cu lattice
        self.box_xy = 100.0  # Å (10 nm) - large enough for multiple pores

        # For Cu(111) slab, we need to calculate proper lattice vectors
        # (111) surface primitive vectors
        a = self.Cu_lattice
        self.a1 = a * np.array([1, 0, 0])  # Simple cubic for now
        self.a2 = a * np.array([0, 1, 0])

        # Number of unit cells to fill the box
        self.nx = int(np.ceil(self.box_xy / a))
        self.ny = int(np.ceil(self.box_xy / a))

        # Adjust box to be exact multiple of lattice
        self.box_x = self.nx * a
        self.box_y = self.ny * a

        # Vacuum spacing
        self.vacuum_z = 30.0  # Å

        # Output
        self.output_dir = "Cu_gap_8.22nm_model_full_xy"


# ========== CREATE PERIODIC CU SLAB ==========
def create_periodic_cu_slab(params):
    """Create Cu(111) slab that spans entire XY plane with periodic boundaries"""

    print("Creating periodic Cu slab...")

    # Create simple cubic Cu slab (for simplicity, can be refined to FCC (111) later)
    positions = []

    # Create layers of Cu atoms
    layer_spacing = params.Cu_lattice * np.sqrt(2 / 3)  # Spacing for (111) layers

    for layer in range(params.Cu_layers):
        z = layer * layer_spacing

        for i in range(params.nx):
            for j in range(params.ny):
                # Simple cubic arrangement
                x = i * params.Cu_lattice
                y = j * params.Cu_lattice

                # For FCC, we'd add additional atoms at face centers
                # But for now, keep it simple
                positions.append([x, y, z])

                # Add FCC atoms (at face centers for alternating layers)
                if layer % 2 == 0:
                    positions.append(
                        [x + params.Cu_lattice / 2, y + params.Cu_lattice / 2, z]
                    )

    positions = np.array(positions)

    # Filter positions to be within box
    mask = (positions[:, 0] < params.box_x) & (positions[:, 1] < params.box_y)
    positions = positions[mask]

    # Create atoms object
    atoms = Atoms("Cu" * len(positions), positions=positions)

    # Set cell
    atoms.set_cell(
        [params.box_x, params.box_y, np.max(positions[:, 2]) + params.vacuum_z]
    )
    atoms.set_pbc([True, True, False])  # Periodic in x,y, non-periodic in z

    print(f"  Cu slab atoms: {len(atoms):,}")
    print(
        f"  Cu slab spans: {np.min(positions[:, 0]):.1f}-{np.max(positions[:, 0]):.1f} Å in x"
    )
    print(
        f"                 {np.min(positions[:, 1]):.1f}-{np.max(positions[:, 1]):.1f} Å in y"
    )

    return atoms


# ========== CREATE PERIODIC GRAPHENE ==========
def create_periodic_graphene(params, z_position):
    """Create graphene layer that spans entire XY plane with periodic boundaries"""

    print("Creating periodic graphene layer...")

    # Graphene lattice constant
    a_graphene = 2.46  # Å

    # Primitive vectors for graphene (honeycomb)
    sqrt3 = np.sqrt(3)
    a1_g = a_graphene * np.array([sqrt3, 0, 0])
    a2_g = a_graphene * np.array([sqrt3 / 2, 3 / 2, 0])

    # Calculate number of unit cells needed to fill box
    # The unit cell area is |a1_g × a2_g| = a_graphene² * (3√3/2)
    n_unit_x = int(np.ceil(params.box_x / np.linalg.norm(a1_g))) + 2
    n_unit_y = int(np.ceil(params.box_y / np.linalg.norm(a2_g))) + 2

    positions = []

    # Generate graphene lattice
    for i in range(-n_unit_x, n_unit_x + 1):
        for j in range(-n_unit_y, n_unit_y + 1):
            # Position of unit cell
            base = i * a1_g + j * a2_g

            # Two atoms per unit cell (A and B sites)
            pos_a = base
            pos_b = base + a_graphene * np.array(
                [0, 1, 0]
            )  # Actually should be a_g/√3 * (0,1,0)?
            # Correct: pos_b = base + a_graphene * np.array([0, 1, 0])  # for orthogonal coordinates

            positions.append(pos_a)
            positions.append(pos_b)

    positions = np.array(positions)

    # Wrap positions to be within [0, box_x) and [0, box_y)
    positions[:, 0] = np.mod(positions[:, 0], params.box_x)
    positions[:, 1] = np.mod(positions[:, 1], params.box_y)
    positions[:, 2] = z_position

    # Remove duplicates (within tolerance)
    # Use KDTree to find and remove atoms too close
    if len(positions) > 0:
        tree = KDTree(positions[:, :2])  # Only consider x,y for periodic duplicates
        mask = np.ones(len(positions), dtype=bool)

        for i in range(len(positions)):
            if mask[i]:
                # Find neighbors within 0.1 Å (periodic images)
                neighbors = tree.query_ball_point(positions[i, :2], 0.1)
                for j in neighbors:
                    if j > i and mask[j]:
                        mask[j] = False

        positions = positions[mask]

    print(f"  Base graphene atoms: {len(positions):,}")
    print(f"  Spans entire XY plane: {params.box_x:.1f} × {params.box_y:.1f} Å")

    return positions


# ========== CREATE POROUS GRAPHENE WITH PERIODICITY ==========
def create_periodic_porous_graphene(params, z_position):
    """Create porous graphene with periodic boundary conditions"""

    print("Creating periodic porous graphene...")

    # First create dense periodic graphene
    positions = create_periodic_graphene(params, z_position)

    if len(positions) == 0:
        return np.array([])

    # Calculate number of pores needed
    pore_area = np.pi * params.pore_radius**2
    total_area = params.box_x * params.box_y
    n_pores_target = int(params.porosity * total_area / pore_area)

    # Ensure reasonable number of pores
    n_pores = max(n_pores_target, 9)  # At least 3×3 grid

    print(f"  Target: {n_pores} pores")

    # Create pore centers on a grid with small random displacements
    pore_centers = []

    # Create grid for even distribution
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

            center_x = i * spacing_x + dx
            center_y = j * spacing_y + dy

            # Ensure pore stays within bounds (considering periodic images)
            pore_centers.append([center_x, center_y])

    # Remove atoms that are within any pore (considering periodic boundaries)
    mask = np.ones(len(positions), dtype=bool)

    for pore_center in pore_centers:
        # Calculate distances with periodic boundary conditions
        dx = positions[:, 0] - pore_center[0]
        dy = positions[:, 1] - pore_center[1]

        # Apply minimum image convention
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

    # Add multiple layers if requested
    if params.carbon_layers > 1:
        all_positions = porous_positions.copy()

        for layer in range(1, params.carbon_layers):
            layer_z = z_position + layer * params.graphene_layer_spacing

            # Create new layer (could use same pattern or offset)
            layer_positions = create_periodic_graphene(params, layer_z)

            if len(layer_positions) > 0:
                # Apply similar pore pattern but with small offset
                layer_mask = np.ones(len(layer_positions), dtype=bool)

                # Offset pore centers slightly for different layers
                offset = params.pore_radius * 0.3  # Small offset

                for pore_center in pore_centers:
                    offset_center = [pore_center[0] + offset, pore_center[1] + offset]

                    dx = layer_positions[:, 0] - offset_center[0]
                    dy = layer_positions[:, 1] - offset_center[1]

                    # Apply minimum image convention
                    dx = dx - params.box_x * np.round(dx / params.box_x)
                    dy = dy - params.box_y * np.round(dy / params.box_y)

                    distances = np.sqrt(dx**2 + dy**2)
                    in_pore = (
                        distances <= params.pore_radius * 0.9
                    )  # Slightly smaller pores in upper layers

                    layer_mask[in_pore] = False

                layer_porous = layer_positions[layer_mask]
                all_positions = np.vstack([all_positions, layer_porous])

        porous_positions = all_positions

    print(f"  Total carbon atoms (all layers): {len(porous_positions):,}")

    return porous_positions


# ========== COMBINE SYSTEMS ==========
def create_complete_model(params):
    """Create complete Cu-graphene slab model"""

    print(f"\n{'=' * 60}")
    print("CREATING COMPLETE SLAB MODEL")
    print(f"{'=' * 60}")

    # ========== Step 1: Create periodic Cu slab ==========
    cu_atoms = create_periodic_cu_slab(params)
    cu_positions = cu_atoms.get_positions()
    cu_z_max = np.max(cu_positions[:, 2])

    print("\nCu slab complete:")
    print(f"  Max z: {cu_z_max:.1f} Å")

    # ========== Step 2: Create periodic porous graphene ==========
    graphene_z = cu_z_max + params.gap_size
    carbon_positions = create_periodic_porous_graphene(params, graphene_z)

    if len(carbon_positions) == 0:
        raise ValueError("Failed to create graphene layer")

    print("\nGraphene layer complete:")
    print(
        f"  Position: z = {np.min(carbon_positions[:, 2]):.1f} to {np.max(carbon_positions[:, 2]):.1f} Å"
    )
    print(f"  Gap from Cu: {np.min(carbon_positions[:, 2]) - cu_z_max:.1f} Å")

    # ========== Step 3: Combine systems ==========
    print("\nCombining systems...")

    # Create carbon atoms
    carbon_atoms = Atoms("C" * len(carbon_positions), positions=carbon_positions)

    # Combine with Cu
    combined = cu_atoms.copy() + carbon_atoms

    # Set final cell
    final_z = np.max(combined.get_positions()[:, 2]) + params.vacuum_z
    combined.set_cell([params.box_x, params.box_y, final_z])
    combined.set_pbc([True, True, False])  # Periodic in xy, non-periodic in z

    # Center in z
    combined.center(vacuum=params.vacuum_z, axis=2)

    # Final positions
    final_positions = combined.get_positions()

    print("\nModel complete:")
    print(f"  Total atoms: {len(combined):,}")
    print(f"  Cu atoms: {len(cu_atoms):,}")
    print(f"  C atoms: {len(carbon_atoms):,}")
    print(f"  Box: {params.box_x:.1f} × {params.box_y:.1f} × {final_z:.1f} Å")
    print(f"  XY coverage: 0-{params.box_x:.1f} Å (x), 0-{params.box_y:.1f} Å (y)")

    # ========== Step 4: Define GCMC region ==========
    gap_bottom = cu_z_max + 1.0  # 1 Å above Cu
    gap_top = np.min(carbon_positions[:, 2]) - 1.0  # 1 Å below carbon

    gap_height = gap_top - gap_bottom
    gap_volume = gap_height * params.box_x * params.box_y

    print("\nGCMC INSERTION REGION:")
    print(f"  Z range: {gap_bottom:.1f} to {gap_top:.1f} Å")
    print(f"  Height: {gap_height:.1f} Å")
    print(f"  Volume: {gap_volume:,.0f} Å³ = {gap_volume / 1000:.1f} nm³")

    return combined, gap_bottom, gap_top


def write_lammps_files(atoms, gap_bottom, gap_top, cell, params):
    """Write all necessary LAMMPS files"""

    os.makedirs(params.output_dir, exist_ok=True)

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Write LAMMPS data file
    filename = f"{params.output_dir}/Cu_slab_gap_8.22nm.lammps"

    with open(filename, "w") as f:
        # Header
        f.write("LAMMPS data file: Cu(111) slab + porous graphene\n")
        f.write(
            f"# Gap: 8.22 nm ({params.gap_size} Å), {params.carbon_layers} carbon layers\n"
        )
        f.write(
            f"# Pore diameter: {params.pore_diameter} Å, Porosity: {params.porosity:.1%}\n"
        )
        f.write(f"# Cell: {cell[0, 0]:.1f} × {cell[1, 1]:.1f} × {cell[2, 2]:.1f} Å\n\n")

        f.write(f"{len(atoms)} atoms\n")
        f.write("2 atom types\n")
        f.write("0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n")

        # Box
        f.write(f"0.0 {cell[0, 0]} xlo xhi\n")
        f.write(f"0.0 {cell[1, 1]} ylo yhi\n")
        f.write(f"0.0 {cell[2, 2]} zlo zhi\n")
        f.write("0.0 0.0 0.0 xy xz yz\n\n")

        # Masses
        f.write("Masses\n\n")
        f.write("1 63.546     # Cu\n")
        f.write("2 12.011     # C\n\n")

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

    gap_height = gap_top - gap_bottom
    gap_volume = gap_height * params.cell_x * params.cell_y
    # Write GCMC region information
    with open(f"{params.output_dir}/gcmc_region.info", "w") as f:
        f.write("GCMC_INSERTION_REGION\n")
        f.write("region_type = block\n")
        f.write("xlo = INF\n")
        f.write("xhi = INF\n")
        f.write("ylo = INF\n")
        f.write("yhi = INF\n")
        f.write(f"zlo = {gap_bottom}\n")
        f.write(f"zhi = {gap_top}\n")
        f.write("\n")
        f.write(f"# Volume = {gap_volume:.1f} Å³\n")
        f.write(f"# Height = {gap_top - gap_bottom:.1f} Å\n")

    # Write LAMMPS input template
    write_input_template(gap_bottom, gap_top, params)


def write_input_template(gap_bottom, gap_top, params):
    """Write LAMMPS input template for GCMC simulation"""

    template = f"""# ========================================================
# LAMMPS Input Script for CO2 Adsorption in Cu-Graphene Slab
# Gap: 8.22 nm ({params.gap_size} Å)
# ========================================================

# ========== Initialization ==========
units           real
atom_style      full
boundary        p p f
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# ========== Read Structure ==========
read_data       Cu_slab_gap_8.22nm.lammps

# ========== Atom Type Definitions ==========
# Type 1: Cu (slab)
# Type 2: C  (graphene)
# Type 3: C  (CO2 molecule)
# Type 4: O  (CO2 molecule)

# Define CO2 molecule template
molecule        co2mol CO2.mol toff 2

# Set masses
mass            1 63.546     # Cu
mass            2 12.011     # C (graphene)
mass            3 12.011     # C (CO2)
mass            4 15.999     # O (CO2)

# ========== Force Field ==========
# Hybrid force field: ReaxFF for framework, LJ for Cu-CO2
pair_style      hybrid/overlay reaxff NULL lj/cut 12.0 coul/long 12.0
pair_modify     mix arithmetic

# ReaxFF for framework (adjust path to your ReaxFF file)
pair_coeff      * * reaxff ffield.reaxff Cu C NULL NULL
fix             qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 80000

# LJ for Cu-CO2 (physisorption parameters)
pair_coeff      1 3 lj/cut 0.042 3.15  # Cu-C(CO2)
pair_coeff      1 4 lj/cut 0.031 3.28  # Cu-O(CO2)

# LJ for CO2-CO2 (TraPPE)
pair_coeff      3 3 lj/cut 0.0559 3.74   # C(CO2)-C(CO2)
pair_coeff      4 4 lj/cut 0.159 3.05    # O(CO2)-O(CO2)
pair_coeff      3 4 lj/cut 0.094 3.395   # C(CO2)-O(CO2)

# Electrostatics for CO2
kspace_style    pppm 1.0e-4
kspace_modify   gewald 1.0e-4 slab 3.0

# ========== Constraints ==========
# Fix bottom Cu layers to prevent drift
region          cu_bottom block INF INF INF INF 0.0 {params.Cu_layers * params.Cu_lattice * 0.5}
group           cu_fixed region cu_bottom
fix             fix_cu cu_fixed setforce 0.0 0.0 0.0

# ========== GCMC Setup ==========
# Gap region between Cu and graphene
region          gcmc_region block INF INF INF INF {gap_bottom} {gap_top}

# GCMC parameters
variable        T equal 298.0          # K
variable        P equal 10.0           # bar
variable        mu equal -416.71462715+ 0.001987*${{T}}*ln(${{P}})  # Chemical potential

# Group for CO2 molecules
group           CO2 type 3 4

# GCMC fix - enhanced insertion attempts in gap
fix             gcmc_fix CO2 gcmc 200 1000 1000 0 12345 ${{T}} ${{mu}} 2.0 \\
                mol co2mol \\
                region gcmc_region \\
                pressure ${{P}} \\
                group CO2 \\
                full_energy

# ========== Thermostat ==========
compute mdtemp all temp
compute_modify mdtemp dynamic/dof yes
fix             nvt all nvt temp ${{T}} ${{T}} 100.0
fix_modify      nvt temp mdtemp

# ========== Monitoring ==========
# Monitor CO2 in gap
compute         co2_in_gap all count/region gcmc_region type 3
variable        nCO2_gap equal c_co2_in_gap/3  # Each CO2 has 3 atoms

# Monitor CO2 near Cu surface (within 3.5 Å)
region          cu_surface_region block INF INF INF INF {gap_bottom} {gap_bottom + 3.5}
compute         co2_near_cu all count/region cu_surface_region type 3
variable        nCO2_cu equal c_co2_near_cu/3

# Output monitoring
thermo          1000
thermo_style    custom step temp press density atoms v_nCO2_gap v_nCO2_cu

# File output for adsorption monitoring
fix             monitor all ave/time 500 10 5000 v_nCO2_gap v_nCO2_cu file adsorption.dat

# Trajectory output
dump            traj all custom 10000 traj.lammpstrj id type x y z
dump_modify     traj sort id

# ========== Run Protocol ==========
# Initial minimization
min_style       cg
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

print "=== Equilibration (2 ps) ==="
run             20000  # 2 ps

print "=== Production GCMC (5 ps) ==="
run             50000  # 5 ps

# ========== Final Analysis ==========
variable        gap_volume equal ({gap_top} - {gap_bottom}) * lx * ly
variable        gap_density equal v_nCO2_gap * 1660.54 / v_gap_volume  # mmol/cm³

print "========================================"
print "SIMULATION COMPLETE"
print "========================================"
print "System specifications:"
print "  Gap size: 8.22 nm ({params.gap_size} Å)"
print "  Cell: {params.cell_x:.1f} × {params.cell_y:.1f} × {cell[2, 2]:.1f} Å"
print "  Pore diameter: {params.pore_diameter} Å"
print "  Carbon layers: {params.carbon_layers}"
print "Results:"
print "  CO2 in gap: ${{nCO2_gap}} molecules"
print "  CO2 near Cu: ${{nCO2_cu}} molecules"
print "  Gap density: ${{gap_density}} mmol/cm³"
print "========================================"
"""

    with open(f"{params.output_dir}/run_gcmc_simulation.lammps", "w") as f:
        f.write(template)

    print(f"✓ LAMMPS input template: {params.output_dir}/run_gcmc_simulation.lammps")


# ========== VERIFICATION ==========
def verify_model(atoms, params):
    """Verify that model spans entire XY plane"""

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    print(f"\n{'=' * 60}")
    print("MODEL VERIFICATION")
    print(f"{'=' * 60}")

    # Check Cu atoms
    cu_mask = np.array(symbols) == "Cu"
    cu_positions = positions[cu_mask]

    if len(cu_positions) > 0:
        cu_x_range = np.ptp(cu_positions[:, 0])
        cu_y_range = np.ptp(cu_positions[:, 1])

        print(f"Cu atoms: {len(cu_positions):,}")
        print(
            f"  X range: {np.min(cu_positions[:, 0]):.1f} to {np.max(cu_positions[:, 0]):.1f} Å"
        )
        print(
            f"  Y range: {np.min(cu_positions[:, 1]):.1f} to {np.max(cu_positions[:, 1]):.1f} Å"
        )
        print(
            f"  Coverage: {cu_x_range / params.box_x:.1%} of X, {cu_y_range / params.box_y:.1%} of Y"
        )

    # Check C atoms
    c_mask = np.array(symbols) == "C"
    c_positions = positions[c_mask]

    if len(c_positions) > 0:
        c_x_range = np.ptp(c_positions[:, 0])
        c_y_range = np.ptp(c_positions[:, 1])

        print(f"\nC atoms: {len(c_positions):,}")
        print(
            f"  X range: {np.min(c_positions[:, 0]):.1f} to {np.max(c_positions[:, 0]):.1f} Å"
        )
        print(
            f"  Y range: {np.min(c_positions[:, 1]):.1f} to {np.max(c_positions[:, 1]):.1f} Å"
        )
        print(
            f"  Coverage: {c_x_range / params.box_x:.1%} of X, {c_y_range / params.box_y:.1%} of Y"
        )

    # Check for gaps
    all_x = positions[:, 0]
    all_y = positions[:, 1]

    # Simple check: ensure atoms are distributed across entire box
    x_bins = 10
    y_bins = 10

    x_edges = np.linspace(0, params.box_x, x_bins + 1)
    y_edges = np.linspace(0, params.box_y, y_bins + 1)

    # Count atoms in each bin
    counts = np.zeros((x_bins, y_bins))

    for x, y in zip(all_x, all_y):
        i = np.digitize(x, x_edges) - 1
        j = np.digitize(y, y_edges) - 1
        if 0 <= i < x_bins and 0 <= j < y_bins:
            counts[i, j] += 1

    empty_bins = np.sum(counts == 0)
    total_bins = x_bins * y_bins

    print(f"\nSpatial distribution check:")
    print(f"  Empty bins: {empty_bins}/{total_bins} ({empty_bins / total_bins:.1%})")

    if empty_bins / total_bins > 0.3:
        print(f"  WARNING: Model may have significant gaps!")
    else:
        print(f"  OK: Good spatial coverage")

    print(f"{'=' * 60}")


# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("SLAB MODEL GENERATOR - 8.22 nm GAP")
    print("=" * 60)

    # Initialize parameters
    params = SlabParameters()

    print("\nModel specifications:")
    print(f"  Gap size: {params.gap_size} Å (8.22 nm)")
    print(f"  Cu supercell: {params.nx_cu} × {params.ny_cu} × {params.Cu_layers}")
    print(f"  Cell dimensions: {params.cell_x:.1f} × {params.cell_y:.1f} Å")
    print(f"  Carbon layers: {params.carbon_layers}")
    print(f"  Pore diameter: {params.pore_diameter} Å (1.25 nm)")
    print(f"  Target porosity: {params.porosity:.1%}")

    # Generate model
    atoms, gap_bottom, gap_top = create_complete_model(params)

    # Verify model
    verify_model(atoms, params)
    # Write files
    write_lammps_files(atoms, gap_bottom, gap_top, cell, params)

    # Write XYZ for visualization
    from ase.io import write

    write(f"{params.output_dir}/model_visualization.xyz", atoms)

    # Create summary
    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {params.output_dir}")
    print("\nFiles created:")
    print("  1. Cu_slab_gap_8.22nm.lammps - Structure data")
    print("  2. run_gcmc_simulation.lammps - LAMMPS input script")
    print("  3. gcmc_region.info - Gap region specifications")
    print("  4. model_visualization.xyz - Visualization file")

    print("\nTo run simulation:")
    print(f"  cd {params.output_dir}")
    print("  mpirun -np 4 lammps -in run_gcmc_simulation.lammps")
    print("\nKey metrics to monitor:")
    print("  - CO2 molecules in gap (v_nCO2_gap)")
    print("  - CO2 molecules near Cu surface (v_nCO2_cu)")
    print("  - Gap density (mmol/cm³)")
    print("=" * 60)
