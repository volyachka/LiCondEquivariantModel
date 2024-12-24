"""
Test for noise sampling in the dataset and property prediction model.
This test checks if the noise standard deviation in the data generation
matches the expected value after sampling noise.
"""

import numpy as np
from torch_geometric.loader import DataLoader
from src.modules.dataset import build_dataset
from src.modules.property_prediction import SevenNetPropertiesPredictor
from src.modules.dataset import AtomsToGraphCollater


def test_noise_sampling():
    """
    Test noise sampling functionality.
    This function tests the noise generation and checks if the standard deviation
    of the noise aligns with the expected value.
    """
    # Build the dataset using the build_dataset function
    dataset = build_dataset()

    # Define the checkpoint name
    checkpoint_name = "7net-0"

    # Initialize the predictor with the checkpoint
    sevennet_predictor = SevenNetPropertiesPredictor(checkpoint_name)

    # Define possible noise levels
    noises = [0.01, 0.1]

    for std_true in noises:
        # Create DataLoader for the dataset
        dataloader = DataLoader(dataset[:10], batch_size=1)
        dataloader.collate_fn = AtomsToGraphCollater(
            cutoff=5, noise_std=std_true, properties_predictor=sevennet_predictor
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
