"""
Utility tests for SevenNet properties predictor and related components.

This module provides pytest fixtures to set up instances of
SevenNetPropertiesPredictor, SevenNetCalculator, and DataLoader for testing.
"""

import pytest
from sevenn.sevennet_calculator import SevenNetCalculator
from torch_geometric.loader import DataLoader
from modules.dataset import build_dataset
from modules.property_prediction import SevenNetPropertiesPredictor


@pytest.fixture
def sevennet_predictor():
    """
    Fixture to initialize and return a SevenNetPropertiesPredictor instance.

    Returns:
        SevenNetPropertiesPredictor: Initialized predictor object.
    """
    predictor_config = {}
    predictor_config["batch_size"] = 50
    predictor_config["checkpoint"] = "7net-0"
    device = "cpu"

    return SevenNetPropertiesPredictor(device=device, predictor_config=predictor_config)


@pytest.fixture
def sevennet_calc():
    """
    Fixture to initialize and return a SevenNetCalculator instance.

    Returns:
        SevenNetCalculator: Initialized calculator object.
    """
    checkpoint_name = "7net-0"
    device = "cpu"

    return SevenNetCalculator(checkpoint_name=checkpoint_name, device=device)


@pytest.fixture
def dataloader():
    """
    Fixture to build a dataset and return a DataLoader instance.

    Returns:
        DataLoader: DataLoader instance for the dataset.
    """
    dataset = build_dataset(
        csv_path="data/sevennet_slopes.csv",
        li_column="v1_Li_slope",
        temp=1000,
        clip_value=0.0001,
        cutoff=5,
    )

    batch_size = 10
    return DataLoader(dataset, batch_size=batch_size)
