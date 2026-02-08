import numpy as np
from ase import Atoms
from ase.io import write
from ase.lattice.cubic import FaceCenteredCubic
from scipy.spatial import KDTree
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import math
import warnings

warnings.filterwarnings("ignore")

# ========== Parameters (in Angstroms) ==========
# Core: Cu nanoparticle (scaled down)
Cu_radius = 12.5  # 2.5 nm = 25 Å
Cu_lattice_constant = 3.615  # Å

# Shell: Porous carbon
gap = 82.2  # 8.22 nm = 82.2 Å
shell_thickness = 2.50  # reduced thickness to save computational cost
inner_radius = Cu_radius + gap  # 107.2 Å
outer_radius = inner_radius + shell_thickness  # 127.2 Å

# Carbon parameters
C_density = 2.0  # g/cm³
C_atomic_radius = 0.77  # Å (covalent radius)
C_min_distance = 1.54  # Å (C-C bond length)
C_min_distance_tolerance = 0.9  # Allow 10% closer than ideal

# Cu-Cu parameters
Cu_atomic_radius = 1.28  # Å (atomic radius)
Cu_min_distance = 2.56  # Å (2 * atomic radius for touching spheres)

# Cu-C parameters
Cu_C_min_distance = Cu_atomic_radius + C_atomic_radius  # 2.05 Å

# Pore parameters
pore_diameter_mean = 12.5  # Å (1.25 nm)
pore_diameter_std = 1.25  # Å
porosity_target = 0.60  # 30% porosity


# ========== Functions for Contact Checking ==========
def check_contacts_parallel(positions1, positions2, min_distance, n_jobs=-1):
    """Check for contacts between two sets of positions using parallel KDTree"""
    if len(positions1) == 0 or len(positions2) == 0:
        return np.array([], dtype=bool)

    print(
        f"  Checking contacts between {len(positions1)} and {len(positions2)} atoms..."
    )

    # Build KDTree for positions2
    tree2 = KDTree(positions2)

    # Process in chunks for memory efficiency
    chunk_size = 5000
    chunks = []
    for i in range(0, len(positions1), chunk_size):
        chunks.append(positions1[i : i + chunk_size])

    def process_chunk(chunk):
        distances, indices = tree2.query(chunk, k=1, workers=1)
        return distances < min_distance

    # Process chunks in parallel
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_chunk)(chunk)
        for chunk in tqdm(chunks, desc="Contact checking", leave=False)
    )

    return np.concatenate(results)


def remove_close_contacts(positions, min_distance, removal_strategy="random"):
    """
    Remove atoms that are too close to each other
    removal_strategy: 'random', 'keep_first', 'grid_based'
    """
    if len(positions) == 0:
        return positions.copy(), 0

    print(f"  Removing close contacts from {len(positions)} atoms...")

    positions_array = positions.copy()
    n_initial = len(positions_array)

    # Use spatial grid for efficient neighbor finding
    grid_size = min_distance * 1.5

    # Build spatial grid
    grid = {}
    atom_indices_in_grid = {}

    for i, pos in enumerate(positions_array):
        cell = tuple((pos // grid_size).astype(int))
        if cell not in grid:
            grid[cell] = []
            atom_indices_in_grid[cell] = []
        grid[cell].append(pos)
        atom_indices_in_grid[cell].append(i)

    # Find atoms to remove
    to_remove = set()

    for cell, cell_atoms in tqdm(grid.items(), desc="Checking cells", leave=False):
        cell_atoms_array = np.array(cell_atoms)
        cell_indices = atom_indices_in_grid[cell]

        # Check within cell
        if len(cell_atoms) > 1:
            tree = KDTree(cell_atoms_array)
            pairs = tree.query_pairs(min_distance * C_min_distance_tolerance)
            for i, j in pairs:
                idx1, idx2 = cell_indices[i], cell_indices[j]
                if removal_strategy == "random":
                    # Randomly choose one to remove
                    if np.random.random() < 0.5:
                        to_remove.add(idx1)
                    else:
                        to_remove.add(idx2)
                else:  # 'keep_first'
                    to_remove.add(idx2)  # Remove the later one

        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if neighbor_cell in grid and (dx != 0 or dy != 0 or dz != 0):
                        neighbor_atoms = np.array(grid[neighbor_cell])
                        neighbor_indices = atom_indices_in_grid[neighbor_cell]

                        # Find pairs between cell and neighbor cell
                        for i, atom1 in enumerate(cell_atoms_array):
                            distances = np.linalg.norm(neighbor_atoms - atom1, axis=1)
                            close_neighbors = np.where(
                                distances < min_distance * C_min_distance_tolerance
                            )[0]

                            for j in close_neighbors:
                                idx1, idx2 = cell_indices[i], neighbor_indices[j]
                                if idx1 not in to_remove and idx2 not in to_remove:
                                    if removal_strategy == "random":
                                        if np.random.random() < 0.5:
                                            to_remove.add(idx1)
                                        else:
                                            to_remove.add(idx2)
                                    else:  # 'keep_first'
                                        to_remove.add(idx2)

    # Remove atoms
    if to_remove:
        print(f"    Found {len(to_remove)} atoms to remove due to close contacts")
        mask = np.ones(len(positions_array), dtype=bool)
        mask[list(to_remove)] = False
        positions_array = positions_array[mask]

    n_removed = n_initial - len(positions_array)
    print(f"    Removed {n_removed} atoms, remaining: {len(positions_array)}")

    return positions_array, n_removed


def analyze_contact_distances(positions, min_distance, atom_type="C-C"):
    """Analyze and report contact distance statistics"""
    if len(positions) < 2:
        return

    print(f"\n  Analyzing {atom_type} contact distances...")

    # Use KDTree for fast nearest neighbor search
    tree = KDTree(positions)
    distances, indices = tree.query(positions, k=2)  # k=2: self + nearest neighbor

    # Extract nearest neighbor distances (skip self)
    nn_distances = distances[:, 1]

    # Statistics
    min_dist = np.min(nn_distances)
    max_dist = np.max(nn_distances)
    mean_dist = np.mean(nn_distances)
    median_dist = np.median(nn_distances)

    # Count too close atoms
    too_close = np.sum(nn_distances < min_distance * C_min_distance_tolerance)
    too_far = np.sum(nn_distances > min_distance * 2.0)  # Arbitrary threshold

    print(f"    Distance statistics:")
    print(f"      Minimum: {min_dist:.3f} Å")
    print(f"      Maximum: {max_dist:.3f} Å")
    print(f"      Mean: {mean_dist:.3f} Å")
    print(f"      Median: {median_dist:.3f} Å")
    print(
        f"    Atoms too close (< {min_distance * C_min_distance_tolerance:.3f} Å): {too_close}"
    )
    print(f"    Atoms too far (> {min_distance * 2.0:.3f} Å): {too_far}")

    # Histogram
    hist, bins = np.histogram(nn_distances, bins=20, range=(0, min_distance * 3))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return {
        "distances": nn_distances,
        "min": min_dist,
        "max": max_dist,
        "mean": mean_dist,
        "median": median_dist,
        "too_close": too_close,
        "hist": hist,
        "bin_centers": bin_centers,
    }


# ========== Step 1: Create Cu Core ==========
print("Creating Cu core...")

# Create a large enough fcc Cu lattice
block_size = int(2 * Cu_radius / Cu_lattice_constant) + 4
cu = FaceCenteredCubic(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    size=(block_size, block_size, block_size),
    symbol="Cu",
    latticeconstant=Cu_lattice_constant,
    pbc=(0, 0, 0),
)

# Center the block
cu.set_positions(cu.get_positions() - np.mean(cu.get_positions(), axis=0))

# Cut sphere: keep atoms within radius
pos = cu.get_positions()
dist = np.linalg.norm(pos, axis=1)
indices = np.where(dist <= Cu_radius)[0]
cu_core = cu[indices]
print(f"Cu core: {len(cu_core)} atoms")

# Analyze Cu-Cu contacts
print("\nAnalyzing Cu-Cu contacts...")
cu_stats = analyze_contact_distances(cu_core.positions, Cu_min_distance, "Cu-Cu")

# ========== Step 2: Generate Carbon Shell ==========
print("\nGenerating carbon shell...")

# Calculate expected number of C atoms
shell_volume = 4 / 3 * math.pi * (outer_radius**3 - inner_radius**3)
mass_per_C_atom = 12 / 6.022e23  # g
volume_per_C_atom = mass_per_C_atom / (C_density * 1e-24)  # Å³
n_C_target = int(shell_volume / volume_per_C_atom)
print(f"Target carbon atoms: {n_C_target}")


def generate_carbon_positions_vectorized(n_positions):
    """Generate random positions in spherical shell using vectorized operations"""
    phi = np.random.uniform(0, 2 * math.pi, n_positions)
    costheta = np.random.uniform(-1, 1, n_positions)
    u = np.random.uniform(inner_radius**3, outer_radius**3, n_positions)

    theta = np.arccos(costheta)
    r = u ** (1 / 3)

    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack([x, y, z])


# Generate initial positions
print("Generating initial carbon positions...")
n_initial = int(n_C_target * 1.5)
carbon_positions = generate_carbon_positions_vectorized(n_initial)
print(f"Generated {len(carbon_positions)} initial carbon positions")

# ========== Step 3: Cu-C Distance Checking ==========
print("\n" + "=" * 60)
print("CHECKING Cu-C DISTANCES")
print("=" * 60)

# Check distances to Cu core
print("Checking distances between carbon atoms and Cu core...")
cu_c_positions = cu_core.positions

# Use geometric pre-filtering first
distances_from_origin = np.linalg.norm(carbon_positions, axis=1)
safe_distance = Cu_radius + gap - Cu_C_min_distance
danger_zone_mask = distances_from_origin < safe_distance

print(f"  Atoms automatically safe: {np.sum(~danger_zone_mask):,}")
print(f"  Atoms needing detailed check: {np.sum(danger_zone_mask):,}")

if np.any(danger_zone_mask):
    # Detailed check for atoms in danger zone
    danger_positions = carbon_positions[danger_zone_mask]

    # Build KDTree for Cu atoms
    cu_tree = KDTree(cu_c_positions)

    # Check distances in parallel
    n_jobs = mp.cpu_count()
    chunk_size = 5000
    chunks = []
    for i in range(0, len(danger_positions), chunk_size):
        chunks.append(danger_positions[i : i + chunk_size])

    print(f"Using {n_jobs} CPU cores for parallel distance checking...")

    def check_cu_distance_chunk(chunk):
        distances, _ = cu_tree.query(chunk, k=1, workers=1)
        return distances >= Cu_C_min_distance

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(check_cu_distance_chunk)(chunk)
        for chunk in tqdm(chunks, desc="Cu-C distance check", leave=False)
    )

    danger_mask = np.concatenate(results)

    # Build final mask
    final_mask = np.ones(len(carbon_positions), dtype=bool)
    final_mask[danger_zone_mask] = danger_mask

    carbon_positions = carbon_positions[final_mask]
    print(f"  Removed {np.sum(~final_mask):,} atoms too close to Cu core")
    print(f"  Remaining carbon atoms: {len(carbon_positions):,}")

# Analyze Cu-C distances for remaining atoms
print("\nAnalyzing Cu-C distances for remaining atoms...")
if len(carbon_positions) > 0:
    cu_tree = KDTree(cu_c_positions)
    distances, _ = cu_tree.query(carbon_positions, k=1)

    min_cu_c_dist = np.min(distances)
    mean_cu_c_dist = np.mean(distances)

    print(
        f"  Minimum Cu-C distance: {min_cu_c_dist:.3f} Å (target: {Cu_C_min_distance:.3f} Å)"
    )
    print(f"  Mean Cu-C distance: {mean_cu_c_dist:.3f} Å")
    print(
        f"  Atoms within {Cu_C_min_distance * 1.1:.2f} Å of Cu: {np.sum(distances < Cu_C_min_distance * 1.1)}"
    )

# ========== Step 4: C-C Distance Filtering ==========
print("\n" + "=" * 60)
print("FILTERING C-C DISTANCES")
print("=" * 60)

# First pass: remove obviously too close atoms
print(
    "First pass: Removing atoms with C-C distances < {:.3f} Å".format(
        C_min_distance * C_min_distance_tolerance
    )
)
carbon_positions_filtered, n_removed_cc1 = remove_close_contacts(
    carbon_positions, C_min_distance, removal_strategy="random"
)

print(f"After first C-C filtering: {len(carbon_positions_filtered):,} atoms")

# Analyze C-C distances
if len(carbon_positions_filtered) > 1:
    cc_stats1 = analyze_contact_distances(
        carbon_positions_filtered, C_min_distance, "C-C (after first pass)"
    )

# ========== Step 5: Pore Creation ==========
print("\n" + "=" * 60)
print("CREATING PORES")
print("=" * 60)

if len(carbon_positions_filtered) > 0:
    n_initial = len(carbon_positions_filtered)
    n_remove_target = int(porosity_target * n_initial)

    # Estimate number of pores
    avg_pore_volume = 4 / 3 * math.pi * (pore_diameter_mean / 2) ** 3
    n_pores = int(porosity_target * shell_volume / avg_pore_volume)
    # n_pores = min(n_pores, 50)

    print(f"Creating {n_pores} pores (target: remove {n_remove_target:,} atoms)")

    atoms_removed = 0
    mask = np.ones(len(carbon_positions_filtered), dtype=bool)

    for pore_idx in tqdm(range(n_pores), desc="Creating pores"):
        # Random pore center in shell
        u = np.random.uniform(inner_radius**3, outer_radius**3)
        r = u ** (1 / 3)
        theta = np.random.uniform(0, math.pi)
        phi = np.random.uniform(0, 2 * math.pi)

        pore_center = np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        )

        # Pore diameter
        pore_diameter = np.random.normal(pore_diameter_mean, pore_diameter_std)
        pore_diameter = max(10, min(15, pore_diameter))
        pore_radius = pore_diameter / 2

        # Find atoms to remove
        distances = np.linalg.norm(carbon_positions_filtered - pore_center, axis=1)
        to_remove = distances <= pore_radius

        # Update mask
        mask &= ~to_remove
        atoms_removed += np.sum(to_remove)

        # Early stopping
        if atoms_removed >= n_remove_target:
            print(f"  Early stopping: removed {atoms_removed:,} atoms")
            break

    carbon_positions_after_pores = carbon_positions_filtered[mask]

    print(f"\nPore creation complete:")
    print(f"  Initial atoms: {n_initial:,}")
    print(f"  Atoms removed by pores: {atoms_removed:,}")
    print(f"  Final atoms: {len(carbon_positions_after_pores):,}")
    print(
        f"  Porosity achieved: {atoms_removed / n_initial:.2%} (target: {porosity_target:.0%})"
    )
else:
    carbon_positions_after_pores = np.array([])

# ========== Step 6: POST-PORE CONTACT CHECKING ==========
print("\n" + "=" * 60)
print("POST-PORE CONTACT ANALYSIS AND CLEANUP")
print("=" * 60)

if len(carbon_positions_after_pores) > 0:
    # Step 6a: Analyze C-C distances after pore creation
    print("\nAnalyzing C-C distances after pore creation...")
    if len(carbon_positions_after_pores) > 1:
        cc_stats_post_pore = analyze_contact_distances(
            carbon_positions_after_pores, C_min_distance, "C-C (post-pore)"
        )

    # Step 6b: Remove any atoms that became too close after pore removal
    print("\nRemoving close contacts created by pore removal...")
    carbon_positions_final, n_removed_post_pore = remove_close_contacts(
        carbon_positions_after_pores, C_min_distance, removal_strategy="random"
    )

    # Step 6c: Final C-C distance analysis
    print("\nFinal C-C distance analysis...")
    if len(carbon_positions_final) > 1:
        cc_stats_final = analyze_contact_distances(
            carbon_positions_final, C_min_distance, "C-C (final)"
        )

    # Step 6d: Final Cu-C distance check
    print("\nFinal Cu-C distance check...")
    if len(carbon_positions_final) > 0:
        cu_tree = KDTree(cu_c_positions)
        distances, _ = cu_tree.query(carbon_positions_final, k=1)

        min_dist_final = np.min(distances)
        problematic = np.where(distances < Cu_C_min_distance)[0]

        print(f"  Minimum Cu-C distance: {min_dist_final:.3f} Å")
        print(
            f"  Problematic atoms (Cu-C < {Cu_C_min_distance:.3f} Å): {len(problematic)}"
        )

        # Remove any atoms too close to Cu
        if len(problematic) > 0:
            print(f"  Removing {len(problematic)} atoms too close to Cu...")
            mask = np.ones(len(carbon_positions_final), dtype=bool)
            mask[problematic] = False
            carbon_positions_final = carbon_positions_final[mask]
            print(f"  Remaining carbon atoms: {len(carbon_positions_final):,}")

    print(f"\nFinal carbon shell statistics:")
    print(f"  Total carbon atoms: {len(carbon_positions_final):,}")
    print(f"  Effective porosity: {1 - len(carbon_positions_final) / n_initial:.2%}")

else:
    carbon_positions_final = np.array([])

# ========== Step 7: Create and Save Structure ==========
print("\n" + "=" * 60)
print("CREATING FINAL STRUCTURE")
print("=" * 60)

# Create carbon atoms
if len(carbon_positions_final) > 0:
    print(f"Creating {len(carbon_positions_final):,} carbon atoms...")
    c_shell = Atoms("C" * len(carbon_positions_final), positions=carbon_positions_final)
else:
    c_shell = Atoms()

# Combine with Cu core
print("Merging core and shell...")
combined = cu_core.copy() + c_shell

# Set cell and center
cell_size = 2.5 * outer_radius
combined.set_cell([cell_size, cell_size, cell_size])
combined.center()

print(f"\nTotal atoms: {len(combined):,}")
print(f"  Cu: {len(cu_core):,}")
print(f"  C: {len(c_shell):,}")

# ========== Step 8: Write Detailed Structure Report ==========
print("\nWriting structure report...")

# Calculate distances for final quality check
if len(c_shell) > 1:
    c_tree = KDTree(c_shell.positions)
    c_distances, _ = c_tree.query(c_shell.positions, k=2)
    c_nn_distances = c_distances[:, 1]

    if len(cu_core) > 1:
        cu_tree = KDTree(cu_core.positions)
        cu_distances, _ = cu_tree.query(cu_core.positions, k=2)
        cu_nn_distances = cu_distances[:, 1]

    if len(c_shell) > 0 and len(cu_core) > 0:
        cu_c_distances, _ = cu_tree.query(c_shell.positions, k=1)

with open("structure_quality_report.txt", "w") as f:
    f.write("Cu_cav@C Structure Quality Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"Cu core atoms: {len(cu_core)}\n")
    f.write(f"Carbon shell atoms: {len(c_shell)}\n")
    f.write(f"Total atoms: {len(combined)}\n\n")

    f.write("Geometry Parameters:\n")
    f.write(f"  Cu core radius: {Cu_radius} Å\n")
    f.write(f"  Gap width: {gap} Å\n")
    f.write(f"  Shell inner radius: {inner_radius} Å\n")
    f.write(f"  Shell outer radius: {outer_radius} Å\n")
    f.write(f"  Shell thickness: {shell_thickness} Å\n")
    f.write(f"  Box size: {cell_size} Å\n\n")

    f.write("Distance Statistics:\n")
    if len(c_shell) > 1:
        f.write(f"  C-C minimum distance: {np.min(c_nn_distances):.3f} Å\n")
        f.write(f"  C-C mean distance: {np.mean(c_nn_distances):.3f} Å\n")
        f.write(
            f"  C-C atoms too close (< {C_min_distance * C_min_distance_tolerance:.3f} Å): {np.sum(c_nn_distances < C_min_distance * C_min_distance_tolerance)}\n"
        )

    if len(cu_core) > 1:
        f.write(f"  Cu-Cu minimum distance: {np.min(cu_nn_distances):.3f} Å\n")
        f.write(f"  Cu-Cu mean distance: {np.mean(cu_nn_distances):.3f} Å\n")

    if len(c_shell) > 0 and len(cu_core) > 0:
        f.write(f"  Cu-C minimum distance: {np.min(cu_c_distances):.3f} Å\n")
        f.write(f"  Cu-C mean distance: {np.mean(cu_c_distances):.3f} Å\n")
        f.write(
            f"  Cu-C atoms too close (< {Cu_C_min_distance:.3f} Å): {np.sum(cu_c_distances < Cu_C_min_distance)}\n"
        )

    f.write("\nPorosity:\n")
    f.write(f"  Target porosity: {porosity_target:.1%}\n")
    f.write(
        f"  Achieved porosity: {1 - len(c_shell) / n_initial if n_initial > 0 else 0:.1%}\n"
    )
    f.write(f"  Pores created: {n_pores}\n")
    f.write(f"  Atoms removed by pores: {atoms_removed}\n")

print("Structure quality report saved to 'structure_quality_report.txt'")

# ========== Step 9: Write LAMMPS Data File ==========
print("\nWriting LAMMPS data file (atom_style full)...")

positions = combined.get_positions()
symbols = combined.get_chemical_symbols()

# Assign molecule IDs (0 for all framework atoms)
molecule_ids = np.zeros(len(combined), dtype=int)

with open(f"Cu_Cav_at_C_checked_{Cu_radius}.lammps", "w") as f:
    # Header
    f.write("LAMMPS data file for Cu_cav@C structure\n")
    f.write("# Generated with comprehensive contact checking\n\n")
    f.write(f"{len(combined)} atoms\n")
    f.write(f"2 atom types\n")
    f.write(f"0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n")

    # Box
    f.write(f"0.0 {cell_size} xlo xhi\n")
    f.write(f"0.0 {cell_size} ylo yhi\n")
    f.write(f"0.0 {cell_size} zlo zhi\n\n")

    # Masses
    f.write("Masses\n\n")
    f.write(f"1 63.546     # Cu\n")
    f.write(f"2 12.011     # C\n\n")

    # Atoms
    f.write("Atoms # full\n\n")

    chunk_size = 10000
    for i in tqdm(range(0, len(combined), chunk_size), desc="Writing atoms"):
        chunk_end = min(i + chunk_size, len(combined))
        chunk_lines = []

        for j in range(i, chunk_end):
            atom_id = j + 1
            mol_id = molecule_ids[j]
            atom_type = 1 if symbols[j] == "Cu" else 2
            charge = 0.0
            pos = positions[j]
            chunk_lines.append(
                f"{atom_id} {mol_id} {atom_type} {charge:.6f} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
            )

        f.write("\n".join(chunk_lines) + "\n")

print(f"\nDone! File saved as 'Cu_Cav_at_C_checked_{Cu_radius}.lammps'")

# ========== Step 10: Visualization Script ==========
print("\nCreating visualization script...")

viz_script = """#!/usr/bin/env python3
"""
# Add visualization script content here

print("\n" + "=" * 60)
print("STRUCTURE GENERATION COMPLETE")
print("=" * 60)
print(f"Cu core diameter: {2 * Cu_radius / 10:.1f} nm")
print(f"Gap (core to shell): {gap / 10:.2f} nm")
print(f"Shell thickness: {shell_thickness / 10:.1f} nm")
print(f"Total atoms: {len(combined):,}")
print(f"  Cu atoms: {len(cu_core):,}")
print(f"  C atoms: {len(c_shell):,}")
print(f"Box size: {cell_size / 10:.1f} nm³")
print(
    f"Minimum Cu-C distance: {np.min(cu_c_distances) if 'cu_c_distances' in locals() else 'N/A':.3f} Å"
)
print(
    f"Minimum C-C distance: {np.min(c_nn_distances) if 'c_nn_distances' in locals() else 'N/A':.3f} Å"
)
print("=" * 60)
