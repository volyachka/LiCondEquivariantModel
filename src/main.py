"""
Main module for training and prediction using the SevenNet model.
"""

import os
import argparse

import torch
import yaml


from modules.nn import SimplePeriodicNetwork
from modules.property_prediction import (
    SevenNetPropertiesPredictor,
    LennardJonesPropertiesPredictor,
)
from modules.train import Trainer

from modules.dataset import (
    build_dataset,
    build_dataloaders_from_dataset,
    AtomsToGraphCollater,
)


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

    property_predictor_name = config["property_predictor"]["name"].lower()
    property_config = config["property_predictor"]["property_config"]

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if not config["training"]["device"]
        else config["training"]["device"]
    )

    dataset = build_dataset(
        csv_path=config["data"]["data_path"],
        li_column=config["data"]["target_column"],
        temp=config["data"]["temperature"],
        clip_value=config["data"]["clip_value"],
        cutoff=config["model"]["radial_cutoff"],
    )

    if property_predictor_name == "sevennet":
        predictor = SevenNetPropertiesPredictor(device, property_config)
    elif property_predictor_name == "lennardjones":
        predictor = LennardJonesPropertiesPredictor(device, property_config, dataset)
    else:
        raise NotImplementedError(
            f"Unsupported property_predictor: {property_predictor_name}"
        )
    # Build dataloaders
    train_dataloader, val_dataloader = build_dataloaders_from_dataset(
        dataset=dataset,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        batch_size=config["data"]["batch_size"],
    )

    # Customize collate functions for dataloaders
    train_dataloader.collate_fn = AtomsToGraphCollater(
        dataset=dataset,
        cutoff=config["model"]["radial_cutoff"],
        noise_std=config["data"]["noise_std"],
        properties_predictor=predictor,
        forces_divided_by_mass=config["training"]["forces_divided_by_mass"],
        num_noisy_configurations=config["training"]["num_noisy_configurations"],
        use_displacements=config["training"]["use_displacements"],
        use_energies=config["training"]["use_energies"],
        upd_neigh_style=config["data"]["upd_neigh_style"],
    )

    val_dataloader.collate_fn = AtomsToGraphCollater(
        dataset=dataset,
        cutoff=config["model"]["radial_cutoff"],
        noise_std=config["data"]["noise_std"],
        properties_predictor=predictor,
        forces_divided_by_mass=config["training"]["forces_divided_by_mass"],
        num_noisy_configurations=config["training"]["num_noisy_configurations"],
        use_displacements=config["training"]["use_displacements"],
        use_energies=config["training"]["use_energies"],
        upd_neigh_style=config["data"]["upd_neigh_style"],
    )

    if config["training"]["use_displacements"]:
        irreps_in = "2x1o"
    else:
        irreps_in = "1x1o"

    if config["training"]["use_energies"]:
        irreps_in = irreps_in + "+1x0e"

    if config["model"]["predict_importance"]:
        irreps_out = "2x0e"
    else:
        irreps_out = "1x0e"

    net = SimplePeriodicNetwork(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        max_radius=config["model"]["radial_cutoff"],
        num_neighbors=config["model"]["num_neighbors"],
        pool_nodes=config["model"]["pool_nodes"],
        num_nodes=config["model"]["num_nodes"],
    )

    # Create a Trainer instance and train the model
    trainer = Trainer(net, train_dataloader, val_dataloader, config)
    trainer.train()


if __name__ == "__main__":
    main()
