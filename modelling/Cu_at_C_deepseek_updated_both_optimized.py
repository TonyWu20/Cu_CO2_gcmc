import numpy as np
from ase import Atoms
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
Cu_radius = 175  # 2.5 nm = 25 Å
Cu_lattice_constant = 3.615  # Å

# Shell: Porous carbon
gap = 82.2  # 8.22 nm = 82.2 Å
shell_thickness = 25.0  # reduced thickness to save computational cost
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

# Output options
generate_both_models = True  # Set to True to generate both models
cell_size = 2.5 * outer_radius  # Box size for both models

# Parallelization parameters
N_JOBS = max(1, mp.cpu_count() - 1)  # Leave one core free
CHUNK_SIZE = 1000  # For parallel processing

# Global variables for parallel processing (to avoid pickling issues)
_global_positions = None
_global_symbols = None
_global_atom_types_dict = None
_global_box_size = None
_global_mean_position = None
_global_molecule_ids = None


# ========== Helper Functions for Parallel Processing ==========
def process_chunk_for_writing(
    start_idx,
    chunk_size,
    positions,
    symbols,
    atom_types_dict,
    box_size,
    mean_position,
    molecule_ids,
):
    """Process a chunk of atoms for writing (top-level function for pickling)"""
    chunk_end = min(start_idx + chunk_size, len(positions))
    chunk_lines = []

    for j in range(start_idx, chunk_end):
        atom_id = j + 1
        mol_id = molecule_ids[j]
        atom_type = atom_types_dict[symbols[j]]
        charge = 0.0
        pos = positions[j]
        # Center atoms in the box
        centered_pos = pos + box_size / 2 - mean_position
        chunk_lines.append(
            f"{atom_id} {mol_id} {atom_type} {charge:.6f} {centered_pos[0]:.8f} {centered_pos[1]:.8f} {centered_pos[2]:.8f}"
        )

    return "\n".join(chunk_lines)


def check_contacts_parallel_batch(positions1_batch, tree2, min_distance):
    """Check contacts for a batch of positions"""
    distances, _ = tree2.query(positions1_batch, k=1, workers=1)
    return distances < min_distance


def check_cu_distance_chunk_parallel(chunk, cu_tree, min_distance):
    """Check Cu distances for a chunk of positions"""
    distances, _ = cu_tree.query(chunk, k=1, workers=1)
    return distances >= min_distance


def get_distances_chunk(chunk, tree, k=2):
    """Get distances for a chunk of positions"""
    distances, _ = tree.query(chunk, k=k, workers=1)
    return distances


def process_cell_contacts(
    cell_key,
    cell_atoms,
    cell_indices,
    neighbor_data_list,
    min_distance,
    removal_strategy,
):
    """Process contacts for a single cell"""
    to_remove = set()

    # Check within cell
    if len(cell_atoms) > 1:
        cell_atoms_array = np.array(cell_atoms)
        tree = KDTree(cell_atoms_array)
        pairs = tree.query_pairs(min_distance * C_min_distance_tolerance)
        for i, j in pairs:
            idx1, idx2 = cell_indices[i], cell_indices[j]
            if removal_strategy == "random":
                if np.random.random() < 0.5:
                    to_remove.add(idx1)
                else:
                    to_remove.add(idx2)
            else:
                to_remove.add(idx2)

    # Check neighboring cells
    for neighbor_atoms, neighbor_indices in neighbor_data_list:
        if len(cell_atoms) > 0 and len(neighbor_atoms) > 0:
            cell_atoms_array = np.array(cell_atoms)
            neighbor_atoms_array = np.array(neighbor_atoms)

            # Use vectorized distance calculation
            diff = (
                cell_atoms_array[:, np.newaxis, :]
                - neighbor_atoms_array[np.newaxis, :, :]
            )
            distances = np.linalg.norm(diff, axis=2)
            close_pairs = np.where(distances < min_distance * C_min_distance_tolerance)

            for i, j in zip(close_pairs[0], close_pairs[1]):
                idx1, idx2 = cell_indices[i], neighbor_indices[j]
                if idx1 not in to_remove and idx2 not in to_remove:
                    if removal_strategy == "random":
                        if np.random.random() < 0.5:
                            to_remove.add(idx1)
                        else:
                            to_remove.add(idx2)
                    else:
                        to_remove.add(idx2)

    return to_remove


def create_pore_single(
    pore_idx,
    n_pores,
    shell_volume,
    porosity_target,
    inner_radius,
    outer_radius,
    pore_diameter_mean,
    pore_diameter_std,
    carbon_positions_filtered,
):
    """Create a single pore - for parallel processing"""
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
    if len(carbon_positions_filtered) > 0:
        distances = np.linalg.norm(carbon_positions_filtered - pore_center, axis=1)
        to_remove = distances <= pore_radius
        return to_remove
    else:
        return np.array([], dtype=bool)


def generate_carbon_chunk(chunk_size, inner_radius, outer_radius):
    """Generate a chunk of carbon positions"""
    phi = np.random.uniform(0, 2 * math.pi, chunk_size)
    costheta = np.random.uniform(-1, 1, chunk_size)
    u = np.random.uniform(inner_radius**3, outer_radius**3, chunk_size)

    theta = np.arccos(costheta)
    r = u ** (1 / 3)

    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack([x, y, z])


# ========== Main Parallel Functions ==========
def check_contacts_parallel(positions1, positions2, min_distance, n_jobs=N_JOBS):
    """Check for contacts between two sets of positions using parallel KDTree"""
    if len(positions1) == 0 or len(positions2) == 0:
        return np.array([], dtype=bool)

    print(
        f"  Checking contacts between {len(positions1):,} and {len(positions2):,} atoms..."
    )

    # Build KDTree for positions2
    tree2 = KDTree(positions2)

    # Process in chunks
    chunk_size = max(CHUNK_SIZE, len(positions1) // (n_jobs * 4))
    chunks = [
        positions1[i : i + chunk_size] for i in range(0, len(positions1), chunk_size)
    ]

    print(f"  Processing {len(chunks)} chunks in parallel...")

    # Parallel processing using joblib
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(check_contacts_parallel_batch)(chunk, tree2, min_distance)
        for chunk in tqdm(chunks, desc="Contact checking", leave=False)
    )

    return np.concatenate(results)


def remove_close_contacts_parallel(
    positions, min_distance, removal_strategy="random", n_jobs=N_JOBS
):
    """
    Remove atoms that are too close to each other using parallel processing
    """
    if len(positions) == 0:
        return positions.copy(), 0

    print(f"  Removing close contacts from {len(positions):,} atoms...")

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

    # Convert lists to arrays for faster access
    for cell in grid:
        grid[cell] = np.array(grid[cell])

    # Prepare cell data for parallel processing
    cell_data_list = []

    for cell_key in grid.keys():
        cell_atoms = grid[cell_key]
        cell_indices = atom_indices_in_grid[cell_key]

        # Get neighbor cells
        neighbor_data_list = []
        cx, cy, cz = cell_key
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor_key = (cx + dx, cy + dy, cz + dz)
                    if neighbor_key in grid:
                        neighbor_data_list.append(
                            (grid[neighbor_key], atom_indices_in_grid[neighbor_key])
                        )

        cell_data_list.append((cell_key, cell_atoms, cell_indices, neighbor_data_list))

    # Process cells in parallel
    print(f"  Processing {len(cell_data_list)} cells in parallel...")

    # Use joblib for parallel processing
    all_to_remove = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_cell_contacts)(
            cell_key,
            cell_atoms,
            cell_indices,
            neighbor_data_list,
            min_distance,
            removal_strategy,
        )
        for cell_key, cell_atoms, cell_indices, neighbor_data_list in tqdm(
            cell_data_list, desc="Checking cell contacts", leave=False
        )
    )

    # Combine results
    to_remove = set()
    for cell_removals in all_to_remove:
        to_remove.update(cell_removals)

    # Remove atoms
    if to_remove:
        print(f"    Found {len(to_remove)} atoms to remove due to close contacts")
        mask = np.ones(len(positions_array), dtype=bool)
        mask[list(to_remove)] = False
        positions_array = positions_array[mask]

    n_removed = n_initial - len(positions_array)
    print(f"    Removed {n_removed} atoms, remaining: {len(positions_array):,}")

    return positions_array, n_removed


def analyze_contact_distances_parallel(
    positions, min_distance, atom_type="C-C", n_jobs=N_JOBS
):
    """Analyze and report contact distance statistics using parallel processing"""
    if len(positions) < 2:
        return None

    print(f"\n  Analyzing {atom_type} contact distances...")

    # Use KDTree for fast nearest neighbor search
    tree = KDTree(positions)

    # Process in chunks for memory efficiency
    chunk_size = max(CHUNK_SIZE, len(positions) // (n_jobs * 2))
    chunks = [
        positions[i : i + chunk_size] for i in range(0, len(positions), chunk_size)
    ]

    # Get nearest neighbor distances in parallel
    nn_chunks = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_distances_chunk)(chunk, tree, 2)
        for chunk in tqdm(chunks, desc="Calculating distances", leave=False)
    )

    # Extract nearest neighbor distances (skip self)
    nn_distances = np.concatenate([distances[:, 1] for distances in nn_chunks])

    # Calculate statistics
    min_dist = np.min(nn_distances)
    max_dist = np.max(nn_distances)
    mean_dist = np.mean(nn_distances)
    median_dist = np.median(nn_distances)

    # Count too close atoms
    too_close = np.sum(nn_distances < min_distance * C_min_distance_tolerance)
    too_far = np.sum(nn_distances > min_distance * 2.0)  # Arbitrary threshold

    print("    Distance statistics:")
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
        "too_far": too_far,
        "hist": hist,
        "bin_centers": bin_centers,
    }


def write_lammps_data_parallel(
    filename, atoms, atom_types_dict, box_size, description="", n_jobs=N_JOBS
):
    """Write LAMMPS data file in atom_style full format with parallel processing"""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Assign molecule IDs (0 for all framework atoms)
    molecule_ids = np.zeros(len(atoms), dtype=int)

    # Get unique element types
    unique_elements = list(set(symbols))

    with open(filename, "w") as f:
        # Header
        f.write(f"LAMMPS data file {description}\n")
        f.write("# Generated with comprehensive contact checking\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(unique_elements)} atom types\n")
        f.write("0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n")

        # Box
        f.write(f"0.0 {box_size} xlo xhi\n")
        f.write(f"0.0 {box_size} ylo yhi\n")
        f.write(f"0.0 {box_size} zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        for element in unique_elements:
            atom_type = atom_types_dict[element]
            mass = atom_types_dict.get(f"{element}_mass", 0.0)
            f.write(f"{atom_type} {mass}     # {element}\n")
        f.write("\n")

        # Atoms
        f.write("Atoms # full\n\n")

        chunk_size = max(CHUNK_SIZE, len(atoms) // (n_jobs * 4))
        chunks = list(range(0, len(atoms), chunk_size))

        # Calculate mean position once
        mean_position = np.mean(positions, axis=0)

        # Process chunks in parallel using joblib
        print(f"  Writing {len(atoms):,} atoms to {filename}...")
        chunk_results = Parallel(n_jobs=min(n_jobs, len(chunks)), verbose=0)(
            delayed(process_chunk_for_writing)(
                start_idx,
                chunk_size,
                positions,
                symbols,
                atom_types_dict,
                box_size,
                mean_position,
                molecule_ids,
            )
            for start_idx in tqdm(chunks, desc="Writing chunks", leave=False)
        )

        # Write all chunks
        for chunk_result in tqdm(chunk_results, desc="Writing file", leave=False):
            f.write(chunk_result + "\n")

    print(f"  File saved as '{filename}'")


def generate_carbon_positions_parallel(
    n_positions, inner_radius, outer_radius, n_chunks=None
):
    """Generate random positions in spherical shell using parallel operations"""
    if n_chunks is None:
        n_chunks = N_JOBS * 2

    # Calculate chunk sizes
    chunk_sizes = [n_positions // n_chunks] * n_chunks
    chunk_sizes[-1] += n_positions % n_chunks

    print(f"  Generating {n_positions:,} positions in {n_chunks} parallel chunks...")

    # Generate chunks in parallel
    chunks = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(generate_carbon_chunk)(chunk_size, inner_radius, outer_radius)
        for chunk_size in tqdm(chunk_sizes, desc="Generating positions", leave=False)
    )

    return np.vstack(chunks)


def create_pores_parallel(
    carbon_positions_filtered,
    n_pores,
    shell_volume,
    porosity_target,
    inner_radius,
    outer_radius,
    pore_diameter_mean,
    pore_diameter_std,
    n_jobs=N_JOBS,
):
    """Create pores in parallel"""
    if len(carbon_positions_filtered) == 0:
        return carbon_positions_filtered, 0

    print(f"Creating {n_pores:,} pores in parallel...")

    # Create pores in parallel
    removal_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(create_pore_single)(
            i,
            n_pores,
            shell_volume,
            porosity_target,
            inner_radius,
            outer_radius,
            pore_diameter_mean,
            pore_diameter_std,
            carbon_positions_filtered,
        )
        for i in tqdm(range(n_pores), desc="Creating pores", leave=False)
    )

    # Combine removal masks
    if removal_results:
        # Combine masks using logical OR
        combined_mask = np.zeros(len(carbon_positions_filtered), dtype=bool)
        for mask in tqdm(removal_results, desc="Combining masks", leave=False):
            if len(mask) > 0:
                combined_mask |= mask

        # Apply combined mask
        mask = ~combined_mask
        carbon_positions_after_pores = carbon_positions_filtered[mask]
        atoms_removed = np.sum(combined_mask)
    else:
        carbon_positions_after_pores = carbon_positions_filtered
        atoms_removed = 0

    return carbon_positions_after_pores, atoms_removed


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
print(f"Cu core: {len(cu_core):,} atoms")

# Center Cu core at origin
cu_core.set_positions(
    cu_core.get_positions() - np.mean(cu_core.get_positions(), axis=0)
)

# Analyze Cu-Cu contacts
print("\nAnalyzing Cu-Cu contacts...")
cu_stats = analyze_contact_distances_parallel(
    cu_core.positions, Cu_min_distance, "Cu-Cu", n_jobs=N_JOBS
)

# ========== Step 2: Create Cu Core Only Model ==========
if generate_both_models:
    print("\n" + "=" * 60)
    print("CREATING Cu CORE ONLY MODEL")
    print("=" * 60)

    # Define atom types for core-only model
    core_atom_types = {"Cu": 1, "Cu_mass": 63.546}

    # Create the core-only model in the same box size
    write_lammps_data_parallel(
        filename=f"Cu_Core_only_{Cu_radius}.lammps",
        atoms=cu_core,
        atom_types_dict=core_atom_types,
        box_size=cell_size,
        description="for Cu core only model",
        n_jobs=N_JOBS,
    )

# ========== Step 3: Generate Carbon Shell ==========
print("\n" + "=" * 60)
print("GENERATING CARBON SHELL")
print("=" * 60)

# Calculate expected number of C atoms
shell_volume = 4 / 3 * math.pi * (outer_radius**3 - inner_radius**3)
mass_per_C_atom = 12 / 6.022e23  # g
volume_per_C_atom = mass_per_C_atom / (C_density * 1e-24)  # Å³
n_C_target = int(shell_volume / volume_per_C_atom)
print(f"Target carbon atoms: {n_C_target:,}")

# Generate initial positions
print("Generating initial carbon positions in parallel...")
n_initial = int(n_C_target * 1.5)
carbon_positions = generate_carbon_positions_parallel(
    n_initial, inner_radius, outer_radius
)
print(f"Generated {len(carbon_positions):,} initial carbon positions")

# ========== Step 4: Cu-C Distance Checking ==========
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
    chunk_size = max(CHUNK_SIZE, len(danger_positions) // (N_JOBS * 4))
    chunks = [
        danger_positions[i : i + chunk_size]
        for i in range(0, len(danger_positions), chunk_size)
    ]

    print(f"  Processing {len(chunks)} chunks in parallel using {N_JOBS} cores...")

    # Process chunks in parallel
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(check_cu_distance_chunk_parallel)(chunk, cu_tree, Cu_C_min_distance)
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

    # Process in parallel chunks
    chunk_size = max(CHUNK_SIZE, len(carbon_positions) // (N_JOBS * 2))
    chunks = [
        carbon_positions[i : i + chunk_size]
        for i in range(0, len(carbon_positions), chunk_size)
    ]

    distance_chunks = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(get_distances_chunk)(chunk, cu_tree, 1)
        for chunk in tqdm(chunks, desc="Calculating Cu-C distances", leave=False)
    )

    distances = np.concatenate(
        [dist[:, 0] if dist.ndim > 1 else dist for dist in distance_chunks]
    )

    min_cu_c_dist = np.min(distances)
    mean_cu_c_dist = np.mean(distances)

    print(
        f"  Minimum Cu-C distance: {min_cu_c_dist:.3f} Å (target: {Cu_C_min_distance:.3f} Å)"
    )
    print(f"  Mean Cu-C distance: {mean_cu_c_dist:.3f} Å")
    print(
        f"  Atoms within {Cu_C_min_distance * 1.1:.2f} Å of Cu: {np.sum(distances < Cu_C_min_distance * 1.1):,}"
    )

# ========== Step 5: C-C Distance Filtering ==========
print("\n" + "=" * 60)
print("FILTERING C-C DISTANCES")
print("=" * 60)

# First pass: remove obviously too close atoms
print(
    f"First pass: Removing atoms with C-C distances < {C_min_distance * C_min_distance_tolerance:.3f} Å"
)
carbon_positions_filtered, n_removed_cc1 = remove_close_contacts_parallel(
    carbon_positions, C_min_distance, removal_strategy="random", n_jobs=N_JOBS
)

print(f"After first C-C filtering: {len(carbon_positions_filtered):,} atoms")

# Analyze C-C distances
if len(carbon_positions_filtered) > 1:
    cc_stats1 = analyze_contact_distances_parallel(
        carbon_positions_filtered,
        C_min_distance,
        "C-C (after first pass)",
        n_jobs=N_JOBS,
    )

# ========== Step 6: Pore Creation ==========
print("\n" + "=" * 60)
print("CREATING PORES IN PARALLEL")
print("=" * 60)

if len(carbon_positions_filtered) > 0:
    n_initial = len(carbon_positions_filtered)
    n_remove_target = int(porosity_target * n_initial)

    # Estimate number of pores
    avg_pore_volume = 4 / 3 * math.pi * (pore_diameter_mean / 2) ** 3
    n_pores = int(porosity_target * shell_volume / avg_pore_volume)

    print(
        f"Creating {n_pores:,} pores in parallel (target: remove {n_remove_target:,} atoms)"
    )

    # Create pores in parallel
    carbon_positions_after_pores, atoms_removed = create_pores_parallel(
        carbon_positions_filtered,
        n_pores,
        shell_volume,
        porosity_target,
        inner_radius,
        outer_radius,
        pore_diameter_mean,
        pore_diameter_std,
        n_jobs=N_JOBS,
    )

    print("\nPore creation complete:")
    print(f"  Initial atoms: {n_initial:,}")
    print(f"  Atoms removed by pores: {atoms_removed:,}")
    print(f"  Final atoms: {len(carbon_positions_after_pores):,}")
    print(
        f"  Porosity achieved: {atoms_removed / n_initial:.2%} (target: {porosity_target:.0%})"
    )
else:
    carbon_positions_after_pores = np.array([])
    atoms_removed = 0

# ========== Step 7: POST-PORE CONTACT CHECKING ==========
print("\n" + "=" * 60)
print("POST-PORE CONTACT ANALYSIS AND CLEANUP")
print("=" * 60)

if len(carbon_positions_after_pores) > 0:
    # Step 6a: Analyze C-C distances after pore creation
    print("\nAnalyzing C-C distances after pore creation...")
    if len(carbon_positions_after_pores) > 1:
        cc_stats_post_pore = analyze_contact_distances_parallel(
            carbon_positions_after_pores,
            C_min_distance,
            "C-C (post-pore)",
            n_jobs=N_JOBS,
        )

    # Step 6b: Remove any atoms that became too close after pore removal
    print("\nRemoving close contacts created by pore removal...")
    carbon_positions_final, n_removed_post_pore = remove_close_contacts_parallel(
        carbon_positions_after_pores,
        C_min_distance,
        removal_strategy="random",
        n_jobs=N_JOBS,
    )

    # Step 6c: Final C-C distance analysis
    print("\nFinal C-C distance analysis...")
    if len(carbon_positions_final) > 1:
        cc_stats_final = analyze_contact_distances_parallel(
            carbon_positions_final, C_min_distance, "C-C (final)", n_jobs=N_JOBS
        )

    # Step 6d: Final Cu-C distance check
    print("\nFinal Cu-C distance check...")
    if len(carbon_positions_final) > 0:
        cu_tree = KDTree(cu_c_positions)

        # Process in parallel chunks
        chunk_size = max(CHUNK_SIZE, len(carbon_positions_final) // (N_JOBS * 2))
        chunks = [
            carbon_positions_final[i : i + chunk_size]
            for i in range(0, len(carbon_positions_final), chunk_size)
        ]

        distance_chunks = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(get_distances_chunk)(chunk, cu_tree, 1)
            for chunk in tqdm(chunks, desc="Final Cu-C check", leave=False)
        )

        distances = np.concatenate(
            [dist[:, 0] if dist.ndim > 1 else dist for dist in distance_chunks]
        )

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

    print("\nFinal carbon shell statistics:")
    print(f"  Total carbon atoms: {len(carbon_positions_final):,}")
    print(f"  Effective porosity: {1 - len(carbon_positions_final) / n_initial:.2%}")

else:
    carbon_positions_final = np.array([])

# ========== Step 8: Create and Save Full Structure ==========
print("\n" + "=" * 60)
print("CREATING FULL STRUCTURE (Cu Core + C Shell)")
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

# Center in the box
combined.set_cell([cell_size, cell_size, cell_size])
combined.center()

print(f"\nTotal atoms: {len(combined):,}")
print(f"  Cu: {len(cu_core):,}")
print(f"  C: {len(c_shell):,}")

# ========== Step 9: Write Full Structure LAMMPS Data File ==========
print("\nWriting full structure LAMMPS data file...")

# Define atom types for full structure
full_atom_types = {"Cu": 1, "C": 2, "Cu_mass": 63.546, "C_mass": 12.011}

write_lammps_data_parallel(
    filename=f"Cu_Cav_at_C_checked_{Cu_radius}.lammps",
    atoms=combined,
    atom_types_dict=full_atom_types,
    box_size=cell_size,
    description="for Cu core + C shell model",
    n_jobs=N_JOBS,
)

# ========== Step 10: Write Detailed Structure Report ==========
print("\nWriting structure report...")

# Calculate distances for final quality check
if len(c_shell) > 1:
    c_tree = KDTree(c_shell.positions)

    # Process in parallel
    chunk_size = max(CHUNK_SIZE, len(c_shell.positions) // (N_JOBS * 2))
    chunks = [
        c_shell.positions[i : i + chunk_size]
        for i in range(0, len(c_shell.positions), chunk_size)
    ]

    c_nn_chunks = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(get_distances_chunk)(chunk, c_tree, 2)
        for chunk in tqdm(chunks, desc="C-C distances", leave=False)
    )

    c_nn_distances = (
        np.concatenate([dist[:, 1] for dist in c_nn_chunks])
        if c_nn_chunks
        else np.array([])
    )

if len(cu_core) > 1:
    cu_tree = KDTree(cu_core.positions)

    chunk_size = max(CHUNK_SIZE, len(cu_core.positions) // (N_JOBS * 2))
    chunks = [
        cu_core.positions[i : i + chunk_size]
        for i in range(0, len(cu_core.positions), chunk_size)
    ]

    cu_nn_chunks = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(get_distances_chunk)(chunk, cu_tree, 2)
        for chunk in tqdm(chunks, desc="Cu-Cu distances", leave=False)
    )

    cu_nn_distances = (
        np.concatenate([dist[:, 1] for dist in cu_nn_chunks])
        if cu_nn_chunks
        else np.array([])
    )

if len(c_shell) > 0 and len(cu_core) > 0:
    cu_tree = KDTree(cu_core.positions)

    chunk_size = max(CHUNK_SIZE, len(c_shell.positions) // (N_JOBS * 2))
    chunks = [
        c_shell.positions[i : i + chunk_size]
        for i in range(0, len(c_shell.positions), chunk_size)
    ]

    cu_c_chunks = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(get_distances_chunk)(chunk, cu_tree, 1)
        for chunk in tqdm(chunks, desc="Cu-C distances", leave=False)
    )

    cu_c_distances = (
        np.concatenate([dist[:, 0] if dist.ndim > 1 else dist for dist in cu_c_chunks])
        if cu_c_chunks
        else np.array([])
    )

# Write report
with open("structure_quality_report.txt", "w") as f:
    f.write("Cu_cav@C Structure Quality Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"Generated with parallel execution using {N_JOBS} CPU cores\n\n")
    f.write("Model 1: Cu Core Only\n")
    f.write(f"  Cu atoms: {len(cu_core)}\n")
    f.write(f"  Box size: {cell_size:.2f} Å\n")
    f.write("\nModel 2: Cu Core + C Shell\n")
    f.write(f"  Cu atoms: {len(cu_core)}\n")
    f.write(f"  C atoms: {len(c_shell)}\n")
    f.write(f"  Total atoms: {len(combined)}\n")
    f.write(f"  Box size: {cell_size:.2f} Å\n\n")

    f.write("Geometry Parameters:\n")
    f.write(f"  Cu core radius: {Cu_radius} Å\n")
    f.write(f"  Gap width: {gap} Å\n")
    f.write(f"  Shell inner radius: {inner_radius} Å\n")
    f.write(f"  Shell outer radius: {outer_radius} Å\n")
    f.write(f"  Shell thickness: {shell_thickness} Å\n\n")

    f.write("Distance Statistics (Full Structure):\n")
    if "c_nn_distances" in locals() and len(c_nn_distances) > 0:
        f.write(f"  C-C minimum distance: {np.min(c_nn_distances):.3f} Å\n")
        f.write(f"  C-C mean distance: {np.mean(c_nn_distances):.3f} Å\n")
        f.write(
            f"  C-C atoms too close (< {C_min_distance * C_min_distance_tolerance:.3f} Å): {np.sum(c_nn_distances < C_min_distance * C_min_distance_tolerance)}\n"
        )

    if "cu_nn_distances" in locals() and len(cu_nn_distances) > 0:
        f.write(f"  Cu-Cu minimum distance: {np.min(cu_nn_distances):.3f} Å\n")
        f.write(f"  Cu-Cu mean distance: {np.mean(cu_nn_distances):.3f} Å\n")

    if "cu_c_distances" in locals() and len(cu_c_distances) > 0:
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

# ========== Step 11: Summary ==========
print("\n" + "=" * 60)
print("MODEL GENERATION COMPLETE")
print("=" * 60)
print(f"Generated {2 if generate_both_models else 1} model(s):")
if generate_both_models:
    print(f"  1. Cu_Core_only_{Cu_radius}.lammps (Cu atoms only)")
    print(f"  2. Cu_Cav_at_C_checked_{Cu_radius}.lammps (Cu + C shell)")
else:
    print(f"  1. Cu_Cav_at_C_checked_{Cu_radius}.lammps (Cu + C shell)")

print("\nCommon parameters:")
print(f"  Cu core diameter: {2 * Cu_radius / 10:.1f} nm")
print(f"  Gap (core to shell): {gap / 10:.2f} nm")
print(f"  Shell thickness: {shell_thickness / 10:.1f} nm")
print(f"  Box size: {cell_size / 10:.1f} nm³")

print("\nCore-only model:")
print(f"  Total atoms: {len(cu_core):,}")
print(f"  Cu atoms: {len(cu_core):,}")

print("\nFull structure:")
print(f"  Total atoms: {len(combined):,}")
print(f"  Cu atoms: {len(cu_core):,}")
print(f"  C atoms: {len(c_shell):,}")

if "cu_c_distances" in locals() and len(cu_c_distances) > 0:
    print(f"  Minimum Cu-C distance: {np.min(cu_c_distances):.3f} Å")

if "c_nn_distances" in locals() and len(c_nn_distances) > 0:
    print(f"  Minimum C-C distance: {np.min(c_nn_distances):.3f} Å")

print("\nParallel execution details:")
print(f"  CPU cores used: {N_JOBS}")
print(f"  Chunk size: {CHUNK_SIZE}")

print("=" * 60)
