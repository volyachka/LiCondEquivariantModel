"""
Main module for training and prediction using the SevenNet model.
"""

import os
import argparse

import torch
import yaml

from modules.dataset import AtomsToGraphCollater, build_dataloader_cv
from modules.nn import SimplePeriodicNetwork
from modules.property_prediction import SevenNetPropertiesPredictor
from modules.train import Trainer


def load_config(config_path):
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    """
    Main entry point for the application.
    Parses arguments, loads configuration, and executes the training process.
    """
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if not config["training"]["device"]
        else config["training"]["device"]
    )

    if config["property_predictor"]["name"].lower() == "sevennet":
        checkpoint_name = config["property_predictor"]["checkpoint"]
        sevennet_predictor = SevenNetPropertiesPredictor(checkpoint_name, device)
    else:
        raise ValueError(
            f"Unsupported property predictor: {config['property_predictor']['name']}"
        )

    # Build dataloaders
    train_dataloader, val_dataloader = build_dataloader_cv(config)

    # Customize collate functions for dataloaders
    train_dataloader.collate_fn = AtomsToGraphCollater(
        cutoff=config["model"]["radial_cutoff"],
        noise_std=config["data"]["noise_std"],
        properties_predictor=sevennet_predictor,
        forces_divided_by_mass=config["training"]["forces_divided_by_mass"],
        num_agg=config["training"]["num_agg"],
        pos_shift=config["training"]["pos_shift"],
        energy_shift=config["training"]["energy_shift"],
    )

    val_dataloader.collate_fn = AtomsToGraphCollater(
        cutoff=config["model"]["radial_cutoff"],
        noise_std=config["data"]["noise_std"],
        properties_predictor=sevennet_predictor,
        forces_divided_by_mass=config["training"]["forces_divided_by_mass"],
        num_agg=config["training"]["num_agg"],
        pos_shift=config["training"]["pos_shift"],
        energy_shift=config["training"]["energy_shift"],
    )

    if config["training"]["predict_importance"]:
        irreps_out = "2x0e"
    else:
        irreps_out = "1x0e"

    if config["training"]["pos_shift"]:
        irreps_in = "2x1o"
    else:
        irreps_in = "1x1o"

    net = SimplePeriodicNetwork(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        max_radius=config["model"]["radial_cutoff"],
        num_neighbors=config["model"]["num_neighbors"],
        pool_nodes=config["model"]["pool_nodes"],
    )

    # Create a Trainer instance and train the model
    trainer = Trainer(net, train_dataloader, val_dataloader, config)
    trainer.train()


if __name__ == "__main__":
    main()
