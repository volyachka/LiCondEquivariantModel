# pylint: disable=R0801
"""
This module contains a test for validating the properties (forces and energy)
predicted by the SevenNet model against the values computed by the SevenNetCalculator.
The test ensures that the model's predictions for forces and energy are consistent 
with the calculations from the SevenNetCalculator.
"""

import numpy as np
from torch_geometric.loader import DataLoader

from sevenn.sevennet_calculator import SevenNetCalculator
from modules.property_prediction import SevenNetPropertiesPredictor


def test_sevennet_properties(
    sevennet_predictor: SevenNetPropertiesPredictor,
    dataloader: DataLoader,
    sevennet_calc: SevenNetCalculator,
):  # pylint: disable=R0914
    """
    Test the consistency between the SevenNet model's predicted properties (forces and energy)
    and the properties calculated by the SevenNetCalculator.

    The test adds noise to atomic positions, then compares the forces and energy predicted by
    the SevenNet model against the values calculated by the SevenNetCalculator.
    """

    for batch in dataloader:
        structures = batch.x["atoms"]

        # Add noise to atomic positions
        for atoms in structures:
            positions = atoms.get_positions()
            noise = np.random.normal(loc=0, scale=0.01, size=positions.shape)
            atoms.set_positions(positions + noise)

        # Predict properties using SevenNet model
        properties_batch = sevennet_predictor.predict(structures)

        # Validate forces and energy against the calculator
        for atoms, forces, energy in zip(
            structures, properties_batch["forces"], properties_batch["energy"]
        ):
            atoms.calc = sevennet_calc
            forces_from_calc = atoms.get_forces()
            energy_from_calc = atoms.get_potential_energy()

            # Assert that the predicted and calculated forces and energies match
            assert np.isclose(
                forces_from_calc, forces.detach().numpy(), atol=1e-6, rtol=1e-5
            ).all()
            assert np.isclose(
                energy_from_calc, energy.detach().numpy(), atol=1e-6, rtol=1e-5
            ).all()
