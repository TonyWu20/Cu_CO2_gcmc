import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.cluster import Icosahedron  # Correct module for nanoparticles
from ase.io import write
import random


def create_cu_core(size_nm=35.0):
    """
    Creates a large Cu nanoparticle using ASE's cluster module.
    Note: 35 nm is extremely large (millions of atoms). This is computationally prohibitive.
    This script uses N=200 as an example for a very large particle, but running this
    is not recommended for MD/GCMC. Use a smaller N for practical models.
    """
    # For a 35 nm particle, N needs to be very high (e.g., N=200 or more)
    # This will create millions of atoms. Use with extreme caution or reduce size.
    N_layers = (
        200  # WARNING: This creates a huge system! (~3.3 million atoms for N=200)
    )
    print(
        f"Creating Cu icosahedron with N={N_layers} layers (WARNING: Very large system)."
    )
    # Use the correct lattice constant for Cu (FCC)
    cu_core = Icosahedron("Cu", noshells=N_layers, latticeconstant=3.615)
    print(f"Cu core created with {len(cu_core)} atoms.")

    # Center the cluster
    cu_core.center()
    return cu_core


def create_solid_carbon_shell(inner_radius_nm, outer_radius_nm, density_nm3=120.0):
    """
    Creates a solid spherical carbon shell between inner and outer radii.
    density_nm3: Approximate number of C atoms per nm^3. Value chosen for amorphous carbon.
                 Adjust as needed. Typical values might be 100-150.
    """
    inner_r = inner_radius_nm * 10.0  # Convert to Ang
    outer_r = outer_radius_nm * 10.0  # Convert to Ang
    density_ang3 = density_nm3 / (10.0**3)  # Convert to atoms per Ang^3

    volume_shell = (4 / 3) * np.pi * (outer_r**3 - inner_r**3)
    total_atoms = int(volume_shell * density_ang3)

    print(
        f"Creating solid C shell: Inner R={inner_r}A, Outer R={outer_r}A, Density={density_nm3}/nm^3"
    )
    print(f"Estimated atoms in solid shell: {total_atoms}")

    positions = []
    symbols = []
    count = 0
    max_attempts = total_atoms * 10  # Limit attempts to avoid infinite loop

    while count < total_atoms and max_attempts > 0:
        # Generate random point in cube
        r_vec = np.random.uniform(-outer_r, outer_r, size=3)
        # Check if it's within the spherical shell
        dist = np.linalg.norm(r_vec)
        if inner_r <= dist <= outer_r:
            positions.append(r_vec)
            symbols.append("C")
            count += 1
        max_attempts -= 1

    if count < total_atoms:
        print(
            f"Warning: Could only place {count} atoms, target was {total_atoms}. Increase max_attempts or reduce density/volume."
        )

    carbon_shell_solid = Atoms(symbols, positions=positions)
    print(f"Solid carbon shell created with {len(carbon_shell_solid)} atoms.")
    return carbon_shell_solid


def introduce_pores(
    carbon_shell,
    num_pores=50,
    pore_radius_nm=0.75,
    shell_inner_nm=43.0,
    shell_outer_nm=45.0,
):
    """
    Removes atoms from the carbon shell to create spherical pores.
    This is a model: pores are spherical voids carved out of the solid shell.
    num_pores: Target number of pores to attempt to place.
    pore_radius_nm: Radius of each spherical pore (target ~1-1.5 nm, so using 0.75 nm for diameter ~1.5nm).
    shell_inner_nm, shell_outer_nm: Bounds of the carbon shell to place pores within.
    """
    print(
        f"Introducing {num_pores} spherical pores (R={pore_radius_nm} nm) into the carbon shell..."
    )
    positions = carbon_shell.get_positions()
    symbols = carbon_shell.get_chemical_symbols()

    placed_pores = 0
    max_attempts = num_pores * 10  # Limit attempts per pore

    pore_positions = []  # Store pore centers for visualization/debugging if needed

    for _ in range(num_pores):
        pore_placed = False
        attempts = 0
        while not pore_placed and attempts < max_attempts:
            # Choose a random center for the pore within the shell volume
            r_center = np.random.uniform(
                -(shell_outer_nm * 10 - pore_radius_nm * 10),
                (shell_outer_nm * 10 - pore_radius_nm * 10),
                size=3,
            )

            # Check if the center is within the allowed shell region
            dist_center = np.linalg.norm(r_center)
            if (
                (shell_inner_nm * 10 + pore_radius_nm * 10)
                <= dist_center
                <= (shell_outer_nm * 10 - pore_radius_nm * 10)
            ):
                # Calculate distance from all atoms to this pore center
                dists = np.linalg.norm(positions - r_center, axis=1)
                # Find atoms within the pore radius
                atoms_to_remove = dists <= (
                    pore_radius_nm * 10
                )  # Convert pore radius to Ang

                # Only remove atoms if the pore center is valid (i.e., it overlaps with the shell)
                if np.any(atoms_to_remove):
                    # Remove atoms within the pore radius
                    keep_indices = ~atoms_to_remove
                    positions = positions[keep_indices]
                    symbols = [s for i, s in enumerate(symbols) if keep_indices[i]]
                    pore_positions.append(r_center)
                    pore_placed = True
                    placed_pores += 1
                    # print(f"Placed pore {placed_pores} at {r_center/10.0} nm") # Debug info
            attempts += 1

    if placed_pores < num_pores:
        print(
            f"Warning: Only placed {placed_pores} out of {num_pores} requested pores."
        )

    # Create the final structure with pores
    carbon_shell_porous = Atoms(symbols, positions=positions)
    print(f"Carbon shell after pore introduction has {len(carbon_shell_porous)} atoms.")
    return carbon_shell_porous


def combine_core_shell(cu_core, carbon_shell):
    """
    Combines the Cu core and the carbon shell into a single structure.
    Assumes both are centered at (0,0,0).
    """
    # Both core and shell should be centered at origin from their creation functions
    combined = cu_core.copy()
    combined.extend(carbon_shell)

    # Recenter the combined structure
    combined.center()

    print(f"Combined structure created with {len(combined)} total atoms.")
    return combined


def write_lammps_data(atoms, filename="Cu_cav@C_model_porous.data"):
    """
    Writes the Atoms object to a LAMMPS data file with atom_style full.
    Assigns atom types: 1 = Cu, 2 = C.
    Assigns masses.
    Initializes bonds/angles as placeholders (required for atom_style full).
    """
    # Define atom types and masses
    atom_types = {"Cu": 1, "C": 2}
    masses = {"Cu": 63.546, "C": 12.011}

    # Assign atom types
    types = []
    for atom in atoms:
        types.append(atom_types[atom.symbol])

    n_atoms = len(atoms)
    n_atom_types = len(set(atom_types.values()))
    n_bond_types = 1  # Placeholder
    n_angle_types = 1  # Placeholder

    # Determine box bounds with padding
    positions = atoms.get_positions()
    padding = 5.0
    x_min, y_min, z_min = positions.min(axis=0) - padding
    x_max, y_max, z_max = positions.max(axis=0) + padding

    with open(filename, "w") as f:
        f.write(f"# LAMMPS data file for Cu@C core/shell with pores (~1-1.5nm) \n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_atom_types} atom types\n")
        f.write(f"{n_bond_types} bond types\n")  # Placeholder
        f.write(f"{n_angle_types} angle types\n")  # Placeholder
        f.write(f"\n")

        f.write(f"{x_min:.6f} {x_max:.6f} xlo xhi\n")
        f.write(f"{y_min:.6f} {y_max:.6f} ylo yhi\n")
        f.write(f"{z_min:.6f} {z_max:.6f} zlo zhi\n")
        f.write(f"\n")

        f.write(f"Masses\n\n")
        for symbol, atom_type in sorted(atom_types.items(), key=lambda item: item[1]):
            f.write(f"{atom_type} {masses[symbol]}\n")
        f.write(f"\n")

        f.write(f"Atoms # full\n\n")
        for i, (atom, atom_type) in enumerate(zip(atoms, types), start=1):
            x, y, z = atom.position
            # atom-ID molecule-ID atom-type q x y z (for atom_style full)
            f.write(f"{i} 1 {atom_type} 0.0 {x:.6f} {y:.6f} {z:.6f}\n")
        f.write(f"\n")

        # Bonds section (Placeholder)
        f.write(f"Bonds\n\n")
        c_indices = [i for i, atom in enumerate(atoms, 1) if atom.symbol == "C"]
        if len(c_indices) >= 2:
            f.write(f"# Dummy bond between first two C atoms\n")
            f.write(f"1 1 {c_indices[0]} {c_indices[1]}\n")
        elif len(atoms) >= 2:
            f.write(f"# Dummy bond between atoms 1 and 2\n")
            f.write(f"1 1 1 2\n")
        f.write(f"\n")

        # Angles section (Placeholder)
        f.write(f"Angles\n\n")
        c_indices_for_angle = [
            i for i, atom in enumerate(atoms, 1) if atom.symbol == "C"
        ]
        if len(c_indices_for_angle) >= 3:
            f.write(f"# Dummy angle involving first three C atoms\n")
            f.write(
                f"1 1 {c_indices_for_angle[0]} {c_indices_for_angle[1]} {c_indices_for_angle[2]}\n"
            )
        elif len(atoms) >= 3:
            f.write(f"# Dummy angle between atoms 1, 2, 3\n")
            f.write(f"1 1 1 2 3\n")
        f.write(f"\n")  # Blank line required at the end

    print(f"LAMMPS data file written to {filename}")


# --- Main Execution ---
if __name__ == "__main__":
    # Set random seed for reproducible pore placement
    random.seed(42)

    # 1. Create the Cu core (35 nm - WARNING: This will be huge!)
    # *** CRITICAL: Running this with N=200 will likely crash or take hours/days.
    # *** For testing, reduce N_layers significantly (e.g., N=20 for ~1.7 nm particle).
    cu_core = create_cu_core(size_nm=35.0)  # BEWARE: Creates a massive system

    # 2. Define shell dimensions based on paper description
    cu_radius_nm = 35.0 / 2.0  # 35 nm particle has 17.5 nm radius
    distance_core_shell_nm = 8.22  # From paper
    c_shell_thickness_nm = 2.0  # Assumed thickness

    shell_inner_radius_nm = (
        cu_radius_nm + distance_core_shell_nm
    )  # 17.5 + 8.22 = 25.72 nm
    shell_outer_radius_nm = (
        shell_inner_radius_nm + c_shell_thickness_nm
    )  # 25.72 + 2.0 = 27.72 nm

    print(f"Cu Core Radius: {cu_radius_nm} nm")
    print(f"C Shell Inner Radius: {shell_inner_radius_nm} nm")
    print(f"C Shell Outer Radius: {shell_outer_radius_nm} nm")

    # 3. Create the solid carbon shell
    carbon_shell_solid = create_solid_carbon_shell(
        inner_radius_nm=shell_inner_radius_nm,
        outer_radius_nm=shell_outer_radius_nm,
        density_nm3=120.0,  # Adjust density as needed
    )

    # 4. Introduce pores into the carbon shell based on PDF data (~1-1.5 nm diameter)
    # Aim for pores with radius ~ 0.75 nm (diameter ~ 1.5 nm) or smaller
    target_pore_radius_nm = 0.75  # ~1.5 nm diameter
    num_pores_to_introduce = 10000  # Adjust number of pores as needed
    carbon_shell_porous = introduce_pores(
        carbon_shell_solid,
        num_pores=num_pores_to_introduce,
        pore_radius_nm=target_pore_radius_nm,
        shell_inner_nm=shell_inner_radius_nm,
        shell_outer_nm=shell_outer_radius_nm,
    )

    # 5. Combine the core and the porous shell
    combined_structure = combine_core_shell(cu_core, carbon_shell_porous)

    # 6. Write the combined structure to a LAMMPS data file
    write_lammps_data(combined_structure, filename="Cu_cav@C_model_porous.data")

    # Optional: Write an XYZ file for visualization (WARNING: File will be huge)
    # write('Cu_cav@C_model_porous.xyz', combined_structure) # Comment out for huge systems
    print("LAMMPS data file 'Cu_cav@C_model_porous.data' created.")
    print(
        "WARNING: The system size is extremely large due to the 35 nm Cu core. Consider using a smaller model for simulations."
    )
