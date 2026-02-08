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
Cu_radius = 25  # 2.5 nm = 25 Å
Cu_lattice_constant = 3.615  # Å

# Shell: Porous carbon
gap = 82.2  # 8.22 nm = 82.2 Å
shell_thickness = 20  # 2 nm = 20 Å
inner_radius = Cu_radius + gap  # 107.2 Å
outer_radius = inner_radius + shell_thickness  # 127.2 Å

# Carbon parameters
C_density = 2.0  # g/cm³
C_atomic_radius = 0.77  # Å (covalent radius)
C_min_distance = 1.54  # Å (C-C bond length)

# Pore parameters
pore_diameter_mean = 12.5  # Å (1.25 nm)
pore_diameter_std = 1.25  # Å
porosity_target = 0.30  # 30% porosity

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

# ========== Step 2: FAST Carbon Shell Generation ==========
print("\nGenerating carbon shell...")

# Calculate expected number of C atoms
shell_volume = 4 / 3 * math.pi * (outer_radius**3 - inner_radius**3)
mass_per_C_atom = 12 / 6.022e23  # g
volume_per_C_atom = mass_per_C_atom / (C_density * 1e-24)  # Å³
n_C_target = int(shell_volume / volume_per_C_atom)
print(f"Target carbon atoms: {n_C_target}")


# Generate carbon positions using vectorized approach
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


# Generate initial positions (150% of target to account for filtering)
print("Generating initial carbon positions...")
n_initial = int(n_C_target * 1.5)
carbon_positions = generate_carbon_positions_vectorized(n_initial)
print(f"Generated {len(carbon_positions)} initial carbon positions")

# ========== Step 3: Optimized Cu Distance Checking ==========
print("\n" + "=" * 60)
print("OPTIMIZED Cu DISTANCE CHECKING")
print("=" * 60)

# Build KDTree for Cu core
print("Building KDTree for Cu core...")
cu_tree = KDTree(cu_core.positions)

# Geometric pre-filtering: atoms far from Cu core are automatically safe
print("Applying geometric pre-filter...")
distances_from_origin = np.linalg.norm(carbon_positions, axis=1)
safe_distance = Cu_radius + gap - C_min_distance * 0.5
danger_zone_mask = distances_from_origin < safe_distance

print(f"  Atoms automatically safe: {np.sum(~danger_zone_mask):,}")
print(f"  Atoms needing detailed check: {np.sum(danger_zone_mask):,}")

if np.any(danger_zone_mask):
    # Only check atoms in danger zone
    danger_positions = carbon_positions[danger_zone_mask]

    # Prepare for parallel processing
    n_jobs = mp.cpu_count()
    print(f"Using {n_jobs} CPU cores for parallel distance checking...")

    # Split danger positions into chunks
    chunk_size = 5000
    chunks = []
    for i in range(0, len(danger_positions), chunk_size):
        chunks.append(danger_positions[i : i + chunk_size])

    print(f"Split into {len(chunks)} chunks for parallel processing")

    # Function to process a single chunk
    def process_chunk(chunk):
        distances, _ = cu_tree.query(chunk, k=1, workers=1)
        return distances >= C_min_distance

    # Process chunks in parallel
    print("Processing chunks in parallel...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_chunk)(chunk)
        for chunk in tqdm(chunks, desc="Distance checking")
    )

    # Combine results
    danger_mask = np.concatenate(results)

    # Build final mask
    final_mask = np.ones(len(carbon_positions), dtype=bool)
    final_mask[danger_zone_mask] = danger_mask
    # Atoms in safe zone are automatically True

    # Apply the filter
    carbon_positions_filtered = carbon_positions[final_mask]

    print(f"\nAfter Cu distance filtering:")
    print(f"  Removed {np.sum(~final_mask):,} atoms (too close to Cu)")
    print(f"  Remaining atoms: {len(carbon_positions_filtered):,}")
else:
    print("All atoms automatically safe!")
    carbon_positions_filtered = carbon_positions.copy()

# ========== Step 4: Carbon-Carbon Distance Filtering ==========
print("\nFiltering carbon-carbon distances...")


def spatial_hash_filter(positions, min_dist, grid_size):
    """Fast filtering using spatial hashing"""
    print("Building spatial hash grid...")
    grid = {}
    filtered_positions = []

    for pos in tqdm(positions, desc="Filtering positions"):
        # Calculate grid cell
        cell = tuple((pos // grid_size).astype(int))

        # Check neighboring cells
        skip = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if neighbor_cell in grid:
                        for neighbor_pos in grid[neighbor_cell]:
                            if np.linalg.norm(pos - neighbor_pos) < min_dist:
                                skip = True
                                break
                    if skip:
                        break
                if skip:
                    break
            if skip:
                break

        if not skip:
            filtered_positions.append(pos)
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(pos)

    return np.array(filtered_positions)


# Apply spatial filtering for C-C distances
carbon_positions_filtered = spatial_hash_filter(
    carbon_positions_filtered, C_min_distance, C_min_distance * 1.5
)
print(f"After C-C distance filtering: {len(carbon_positions_filtered):,} atoms")

# ========== Step 5: Efficient Pore Creation ==========
print("\nCreating pores...")

if len(carbon_positions_filtered) > 0:
    n_initial = len(carbon_positions_filtered)
    n_remove_target = int(porosity_target * n_initial)

    # Estimate number of pores needed
    avg_pore_volume = 4 / 3 * math.pi * (pore_diameter_mean / 2) ** 3
    n_pores = int(porosity_target * shell_volume / avg_pore_volume)
    # n_pores = min(n_pores, 50)  # Limit for speed

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
                r * math.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        )

        # Pore diameter with normal distribution
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

    carbon_positions_final = carbon_positions_filtered[mask]
    print(f"\nPore creation complete:")
    print(f"  Final carbon atoms: {len(carbon_positions_final):,}")
    print(f"  Atoms removed by pores: {atoms_removed:,}")
    print(
        f"  Porosity achieved: {atoms_removed / n_initial:.2%} (target: {porosity_target:.0%})"
    )
else:
    carbon_positions_final = np.array([])

# ========== Step 6: Create and Save Structure ==========
print("\nCreating final structure...")

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

# ========== Step 7: Write LAMMPS Data File ==========
print("\nWriting LAMMPS data file...")

# Prepare data
positions = combined.get_positions()
symbols = combined.get_chemical_symbols()

with open("Cu_Cav_at_C_optimized.lammps", "w") as f:
    # Header
    f.write("LAMMPS data file for Cu_cav@C structure\n\n")
    f.write(f"{len(combined)} atoms\n")
    f.write("2 atom types\n\n")

    # Box
    f.write(f"0.0 {cell_size} xlo xhi\n")
    f.write(f"0.0 {cell_size} ylo yhi\n")
    f.write(f"0.0 {cell_size} zlo zhi\n\n")

    # Atoms
    f.write("Atoms\n\n")

    # Write in chunks
    chunk_size = 10000
    for i in tqdm(range(0, len(combined), chunk_size), desc="Writing chunks"):
        chunk_end = min(i + chunk_size, len(combined))
        chunk_lines = []
        for j in range(i, chunk_end):
            atom_type = 1 if symbols[j] == "Cu" else 2
            pos = positions[j]
            chunk_lines.append(
                f"{j + 1} 0 {atom_type} 0.0 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
            )
        f.write("\n".join(chunk_lines) + "\n")

print(f"\nDone! File saved as 'Cu_Cav_at_C_optimized.lammps'")

# ========== Step 8: Summary ==========
print("\n" + "=" * 60)
print("STRUCTURE SUMMARY")
print("=" * 60)
print(f"Cu core diameter: {2 * Cu_radius / 10:.1f} nm")
print(f"Gap (core to shell): {gap / 10:.2f} nm")
print(f"Shell thickness: {shell_thickness / 10:.1f} nm")
print(f"Shell inner radius: {inner_radius / 10:.1f} nm")
print(f"Shell outer radius: {outer_radius / 10:.1f} nm")
print(f"Total atoms: {len(combined):,}")
print(f"  Cu atoms: {len(cu_core):,}")
print(f"  C atoms: {len(c_shell):,}")
print(f"Box size: {cell_size / 10:.1f} nm³")
print("=" * 60)

# Save summary
with open("structure_summary_optimized.txt", "w") as f:
    f.write("Cu_cav@C Structure (Optimized Parallel Generation)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Cu core diameter: {2 * Cu_radius / 10:.1f} nm\n")
    f.write(f"Gap (core to shell): {gap / 10:.2f} nm\n")
    f.write(f"Shell thickness: {shell_thickness / 10:.1f} nm\n")
    f.write(f"Total atoms: {len(combined)}\n")
    f.write(f"  Cu atoms: {len(cu_core)}\n")
    f.write(f"  C atoms: {len(c_shell)}\n")
    f.write(
        f"Carbon density: {len(c_shell) / shell_volume if shell_volume > 0 else 0:.4f} atoms/Å³\n"
    )
    f.write(
        f"Shell porosity: {atoms_removed / n_initial if n_initial > 0 else 0:.2%}\n"
    )
    f.write(f"Box size: {cell_size / 10:.1f} nm\n")
    f.write(f"Parallel processing: {mp.cpu_count()} cores\n")

print("\nSummary saved to 'structure_summary_optimized.txt'")
