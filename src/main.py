"""
Main module for training and prediction using the SevenNet model.
"""

import os
import argparse
from typing import List

# Third-party imports
import torch
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# First-party imports
from models.mixing_network import MixingNetwork
from models.simple_network import SimplePeriodicNetwork

from modules.property_prediction import (
    SevenNetPropertiesPredictor,
    AseCalculatorPropertiesPredictor,
    RandomPropertiesPredictor,
)

from modules.train import Trainer

from modules.dataset import (
    build_dataset,
    build_dataloaders_from_dataset,
    build_superionic_toy_dataset,
    build_dataset_snapshots_by_sevennet,
    build_datasets_with_selected_by_random_samples,
    build_extended_sevennet,
    AtomsToGraphCollater,
)


def load_config(config_path: str):
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


def select_dataset(config: dict):
    """
    Selects and returns a dataset based on the given configuration.
    """
    match config["data"]["name"]:
        case "md_by_sevennet":
            return build_dataset(
                csv_path=config["data"]["data_path"],
                li_column=config["data"]["target_column"],
                temp=config["data"]["temperature"],
                clip_value=config["data"]["clip_value"],
                cutoff=config["model"]["radial_cutoff"],
            )
        case "toy_dataset":
            return build_superionic_toy_dataset(
                root_folder=config["data"]["root_folder"],
                clip_value=config["data"]["clip_value"],
                cutoff=config["model"]["radial_cutoff"],
            )
        case "snapshots_by_sevennet":
            return build_dataset_snapshots_by_sevennet(
                csv_path=config["data"]["data_path"],
                li_column=config["data"]["target_column"],
                temp=config["data"]["temperature"],
                clip_value=config["data"]["clip_value"],
                cutoff=config["model"]["radial_cutoff"],
                skip_first_fs=config["data"]["skip_first_fs"],
                step_size_fs=config["data"]["step_size_fs"],
            )
        case "md_by_sevennet_with_selected_by_random_samples":
            return build_datasets_with_selected_by_random_samples(
                csv_path=config["data"]["data_path"],
                li_column=config["data"]["target_column"],
                temp=config["data"]["temperature"],
                clip_value=config["data"]["clip_value"],
                cutoff=config["model"]["radial_cutoff"],
            )
        case "extended_md_by_sevennet":
            return build_extended_sevennet(
                root_folder=config["data"]["root_folder"],
                clip_value=config["data"]["clip_value"],
                cutoff=config["model"]["radial_cutoff"],
                strategy_sampling=config["training"]["strategy_sampling"],
                skip_first_fs=config["model"].get("skip_first_fs", None),
                step_size_fs=config["model"].get("step_size_fs", None),
            )
        case _:
            raise NotImplementedError()


def select_property_predictor(config: dict, dataset: List[Data], device: str):
    """
    Selects and returns the appropriate property predictor based on the configuration.
    """
    property_predictor_name = config["property_predictor"]["name"].lower()
    property_config = config["property_predictor"]["property_config"]

    match property_predictor_name:
        case "sevennet":
            return SevenNetPropertiesPredictor(device, property_config)
        case "lennardjones" | "superionic_toy":
            return AseCalculatorPropertiesPredictor(
                device, property_predictor_name, property_config, dataset
            )
        case "random":
            return RandomPropertiesPredictor(device)
        case _:
            raise NotImplementedError(
                f"Unsupported property_predictor: {property_predictor_name}"
            )


def select_model(config: dict, irreps_in: str, irreps_out: str):
    """
    Selects and returns the appropriate model based on the configuration.
    """
    if config["model"]["mix_properites"]:
        return MixingNetwork(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            max_radius=config["model"]["radial_cutoff"],
            num_neighbors=config["model"]["num_neighbors"],
            num_nodes=config["model"]["num_nodes"],
            number_of_basis=config["model"]["number_of_basis"],
            mul=config["model"]["mul"],
            layers=config["model"]["layers"],
            lmax=config["model"]["lmax"],
            fc_neurons=config["model"]["fc_neurons"],
            pool_nodes=config["model"]["pool_nodes"],
        )
    return SimplePeriodicNetwork(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        max_radius=config["model"]["radial_cutoff"],
        num_neighbors=config["model"]["num_neighbors"],
        num_nodes=config["model"]["num_nodes"],
        number_of_basis=config["model"]["number_of_basis"],
        mul=config["model"]["mul"],
        layers=config["model"]["layers"],
        lmax=config["model"]["lmax"],
        fc_neurons=config["model"]["fc_neurons"],
        pool_nodes=config["model"]["pool_nodes"],
    )


def main():  # pylint: disable=R0914
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

    dataset = select_dataset(config)
    if config["data"]["name"] == "md_by_sevennet_with_selected_by_random_samples":
        dataset, rnd_dataset = dataset[0], dataset[1]
        rnd_predictor = select_property_predictor(config, rnd_dataset, device)
        rnd_dataloader = DataLoader(rnd_dataset, batch_size=10, shuffle=False)
        rnd_dataloader.collate_fn = AtomsToGraphCollater(
            dataset=rnd_dataset,
            cutoff=config["model"]["radial_cutoff"],
            noise_std=config["data"]["noise_std"],
            properties_predictor=rnd_predictor,
            forces_divided_by_mass=config["training"]["forces_divided_by_mass"],
            num_noisy_configurations=config["training"]["num_noisy_configurations"],
            use_displacements=config["training"]["use_displacements"],
            use_energies=config["training"]["use_energies"],
            upd_neigh_style=config["data"]["upd_neigh_style"],
            predict_per_atom=config["training"]["predict_per_atom"],
            clip_value=config["data"]["clip_value"],
            strategy_sampling=config["training"]["strategy_sampling"],
            node_style_build=config["model"]["node_style_build"],
            device=device,
        )

    predictor = select_property_predictor(config, dataset, device)

    train_dataloader, val_dataloader = build_dataloaders_from_dataset(
        dataset=dataset,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        batch_size=config["data"]["batch_size"],
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
        predict_per_atom=config["training"]["predict_per_atom"],
        clip_value=config["data"]["clip_value"],
        strategy_sampling=config["training"]["strategy_sampling"],
        node_style_build=config["model"]["node_style_build"],
        device=device,
    )

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
        predict_per_atom=config["training"]["predict_per_atom"],
        clip_value=config["data"]["clip_value"],
        strategy_sampling=config["training"]["strategy_sampling"],
        node_style_build=config["model"]["node_style_build"],
        device=device,
    )

    irreps_in = "2x1o" if config["training"]["use_displacements"] else "1x1o"
    if config["training"]["use_energies"]:
        irreps_in += "+1x0e"
    irreps_out = "2x0e" if config["model"]["predict_importance"] else "1x0e"

    val_dataloaders = {"val": val_dataloader}

    if config["data"]["name"] == "md_by_sevennet_with_selected_by_random_samples":
        val_dataloaders["rnd"] = rnd_dataloader
    net = select_model(config, irreps_in, irreps_out)
    trainer = Trainer(net, train_dataloader, val_dataloaders, config)
    trainer.train()


if __name__ == "__main__":
    main()
