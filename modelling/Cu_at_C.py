import numpy as np
from ase import Atoms
from ase.cluster import Decahedron
from ase.io import write
import math

# Constants from the PDF
cu_diameter = 350.0  # 35 nm = 350 Angstrom
gap_distance = 82.2  # 8.22 nm = 82.2 Angstrom
gap_std = 18.5  # 1.85 nm = 18.5 Angstrom
box_padding = 30.0  # Additional padding for the simulation box

# Create a five-fold twinned Cu nanoparticle (using decahedral cluster which has 5-fold symmetry)
cu_lattice_constant = 3.61  # Cu lattice constant in Angstrom
cu_atoms = Decahedron("Cu", latticeconstant=cu_lattice_constant, p=25, q=15, r=10)
cu_atoms.center()

# Scale to the target diameter of 35nm
current_diameter = cu_atoms.get_diameter()
scaling_factor = cu_diameter / current_diameter
cu_atoms.positions *= scaling_factor
cu_atoms.center()

# Create porous carbon shell
cu_radius = cu_diameter / 2
shell_inner_radius = cu_radius + gap_distance
shell_outer_radius = shell_inner_radius + 10.0  # Carbon shell thickness ~1nm

# Generate carbon atoms in spherical shell with minimum separation
carbon_positions = []
min_c_distance = 1.42  # Approximate C-C bond length in Angstrom
max_attempts = 1000000
attempt = 0
num_carbon_target = 2000  # Target number of carbon atoms

while len(carbon_positions) < num_carbon_target and attempt < max_attempts:
    # Generate random position in spherical coordinates
    r = np.random.uniform(shell_inner_radius, shell_outer_radius)
    theta = np.random.uniform(0, 2 * math.pi)
    phi = np.arccos(2 * np.random.uniform(0, 1) - 1)

    # Convert to Cartesian coordinates
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)

    # Check minimum distance to existing carbon atoms
    too_close = False
    for cp in carbon_positions:
        if np.linalg.norm(np.array([x, y, z]) - np.array(cp)) < min_c_distance:
            too_close = True
            break

    if not too_close:
        carbon_positions.append([x, y, z])

    attempt += 1

# Create carbon atoms object
carbon_atoms = Atoms("C" * len(carbon_positions), positions=carbon_positions)
carbon_atoms.center()

# Introduce porosity (remove random atoms)
porosity = 0.6  # 60% porosity
remove_indices = np.random.choice(
    len(carbon_atoms), int(porosity * len(carbon_atoms)), replace=False
)
mask = np.ones(len(carbon_atoms), dtype=bool)
mask[remove_indices] = False
porous_carbon_atoms = carbon_atoms[mask]

# Create larger cavities in the carbon shell
num_cavities = 12
cavity_radius = 15.0  # Angstrom
for _ in range(num_cavities):
    # Random center within shell region
    r = np.random.uniform(shell_inner_radius, shell_outer_radius)
    theta = np.random.uniform(0, 2 * math.pi)
    phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
    cx = r * math.sin(phi) * math.cos(theta)
    cy = r * math.sin(phi) * math.sin(theta)
    cz = r * math.cos(phi)

    # Remove atoms within cavity_radius
    distances = np.linalg.norm(
        porous_carbon_atoms.positions - np.array([cx, cy, cz]), axis=1
    )
    mask = distances > cavity_radius
    porous_carbon_atoms = porous_carbon_atoms[mask]

# Combine Cu core and carbon shell
combined_atoms = cu_atoms + porous_carbon_atoms
combined_atoms.center(vacuum=box_padding)

# Shift all atoms to ensure positive coordinates for LAMMPS
min_coords = combined_atoms.positions.min(axis=0)
combined_atoms.positions -= min_coords

# Create atom types array (1=Cu, 2=C)
atom_types = np.ones(len(combined_atoms), dtype=int)
atom_types[len(cu_atoms) :] = 2
combined_atoms.set_array("atom_types", atom_types)

# Write to LAMMPS data file with atomic style (includes mass information)
write(
    "Cu_at_C.data",
    combined_atoms,
    format="lammps-data",
    atom_style="atomic",
    units="metal",
    masses=True,
)

print(
    f"Created structure with {len(cu_atoms)} Cu atoms and {len(porous_carbon_atoms)} C atoms"
)
print(f"Unit cell dimensions: {combined_atoms.cell.array.diagonal()}")
print("LAMMPS data file 'Cu_at_C.data' has been generated with atom_style='atomic'")
