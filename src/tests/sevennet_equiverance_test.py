"""
This module contains tests for the equivariance properties of the SevenNet model.
The tests check the invariance of energy and forces predictions under rotations
and shifts of atomic structures with added noise. This ensures that the model
preserves symmetry properties relevant to physical systems.
"""

from copy import deepcopy

import numpy as np
from torch_geometric.loader import DataLoader
from e3nn import o3  # Import directly as recommended
from tqdm import tqdm
import torch

from modules.dataset import build_dataset
from modules.property_prediction import SevenNetPropertiesPredictor


def add_noise_and_rotate(atoms, noise_std=0.01):
    """
    Adds noise and applies random rotation and translation to the atomic positions.

    Args:
        atoms (Atoms): Atomic structure object.
        noise_std (float): Standard deviation for Gaussian noise added to atomic positions.

    Returns:
        atoms_rotated (Atoms): Rotated atomic structure object.
        wigner_d (np.ndarray): Rotation matrix.
    """
    positions = atoms.get_positions()
    noise = np.random.normal(loc=0, scale=noise_std, size=positions.shape)

    # Add noise to positions
    atoms.set_positions(positions + noise)
    atoms_rotated = deepcopy(atoms)

    positions = atoms_rotated.get_positions()
    angles = torch.rand(3) * np.pi * 2
    wigner_d = o3.wigner_D(1, *angles).detach().numpy()

    fractional_shift = np.random.rand(1, 3)
    cartesian_shift = np.dot(fractional_shift, atoms.cell)

    # Apply rotation and shift to the positions
    rotated_positions = (positions + cartesian_shift) @ wigner_d.T

    atoms_rotated.cell = atoms_rotated.cell @ wigner_d.T
    atoms_rotated.set_positions(rotated_positions)

    return atoms_rotated, wigner_d


def test_sevennet_equivariance_energies():
    """
    Test the equivariance of energy predictions under rotation and translation.

    This function verifies that the energy prediction by the SevenNet model remains
    invariant under random rotations and translations of atomic structures with
    added noise. It compares the energy of the original and transformed structures.
    """
    dataset = build_dataset(
        csv_path="data/sevennet_slopes.csv",
        li_column="v1_Li_slope",
        temp=1000,
        clip_value=0.0001,
    )

    checkpoint_name = "7net-0"
    batch_size = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the predictor with the checkpoint
    sevennet_predictor = SevenNetPropertiesPredictor(
        checkpoint_name, batch_size, device
    )

    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch in tqdm(dataloader):
        structures = batch.x["atoms"]
        initial_atoms_arr = []
        rotated_atoms_arr = []

        for atoms in structures:
            atoms_rotated, _ = add_noise_and_rotate(atoms)
            initial_atoms_arr.append(atoms)
            rotated_atoms_arr.append(atoms_rotated)

        # Energy check
        np.testing.assert_allclose(
            sevennet_predictor.predict(initial_atoms_arr)["energy"],
            sevennet_predictor.predict(rotated_atoms_arr)["energy"],
            atol=1e-6,
            rtol=1e-6,
        )


def test_sevennet_equivariance_forces():  # pylint: disable=R0914
    """
    Test the equivariance of force predictions under rotation and translation.

    This function verifies that the forces predicted by the SevenNet model remain
    invariant under random rotations and translations of atomic structures with
    added noise. It compares the forces of the original and transformed structures.
    """
    dataset = build_dataset(
        csv_path="data/sevennet_slopes.csv",
        li_column="v1_Li_slope",
        temp=1000,
        clip_value=0.0001,
    )

    checkpoint_name = "7net-0"
    batch_size = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the predictor with the checkpoint
    sevennet_predictor = SevenNetPropertiesPredictor(
        checkpoint_name, batch_size, device
    )

    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch in tqdm(dataloader):
        structures = batch.x["atoms"]
        initial_atoms_arr = []
        rotated_atoms_arr = []
        wigner_matrices_arr = []

        for atoms in structures:
            atoms_rotated, wigner_d = add_noise_and_rotate(atoms)
            initial_atoms_arr.append(atoms)
            rotated_atoms_arr.append(atoms_rotated)
            wigner_matrices_arr.append(wigner_d)

        # Forces check
        for force_initial, force_rotated, wigner_d in zip(
            sevennet_predictor.predict(initial_atoms_arr)["forces"],
            sevennet_predictor.predict(rotated_atoms_arr)["forces"],
            wigner_matrices_arr,
        ):
            np.testing.assert_allclose(
                force_initial @ wigner_d.T, force_rotated, atol=1e-3, rtol=1e-5
            )

        print("All assertions passed")
