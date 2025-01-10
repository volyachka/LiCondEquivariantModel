# pylint: disable=R0801

"""
Test for noise sampling in the dataset and property prediction model.
This test checks if the noise standard deviation in the data generation
matches the expected value after sampling noise.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from modules.dataset import build_dataset
from modules.property_prediction import SevenNetPropertiesPredictor
from modules.dataset import AtomsToGraphCollater


def test_noise_sampling():
    """
    Test noise sampling functionality.
    This function tests the noise generation and checks if the standard deviation
    of the noise aligns with the expected value.
    """
    # Build the dataset using the build_dataset function
    dataset = build_dataset(
        csv_path="data/sevennet_slopes.csv",
        li_column="v1_Li_slope",
        temp=1000,
        clip_value=0.0001,
    )
    # Define the checkpoint name
    checkpoint_name = "7net-0"
    batch_size = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the predictor with the checkpoint
    sevennet_predictor = SevenNetPropertiesPredictor(
        checkpoint_name, batch_size, device
    )
    # Define possible noise levels
    noises = [0.01, 0.1]

    for std_true in noises:
        # Create DataLoader for the dataset
        dataloader = DataLoader(dataset[:10], batch_size=1)

        dataloader.collate_fn = AtomsToGraphCollater(
            cutoff=5,
            noise_std=5,
            properties_predictor=sevennet_predictor,
            forces_divided_by_mass=True,
            num_noisy_configurations=1,
            use_displacements=False,
            use_energies=False,
        )

        noise_arr = []

        # Iterate through the DataLoader
        for i, sample in enumerate(dataloader):
            # Get initial and noisy coordinates
            initial_coords = dataset[i].x["atoms"].get_positions().reshape(-1)
            noise_coords = sample["pos"].reshape(-1)
            noise = (noise_coords - initial_coords).detach().tolist()
            noise_arr.extend(noise)

        # Calculate the standard deviation of the noise
        std_exp = np.array(noise_arr).std()

        # Assert that the experimental and true standard deviations match within a 10% margin
        assert np.abs((std_true - std_exp) / std_true) <= 0.1
