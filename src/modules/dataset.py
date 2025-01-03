"""
Dataset utilities and data processing for the SevenNet model.
"""

# Standard imports
# Standard library imports
from copy import deepcopy
from typing import Any, List, Optional

# Third-party imports
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor

# First-party imports
from modules.utils import query_mpid_structure


def build_dataloader_cv(config):
    """
    Build dataloaders for cross-validation.

    Args:
        config (dict): Configuration dictionary with data and training parameters.

    Returns:
        tuple: Train and validation dataloaders.
    """
    dataset = build_dataset(
        csv_path=config["data"]["data_path"],
        li_column=config["data"]["target_column"],
        temp=config["data"]["temperature"],
        clip_value=config["data"]["clip_value"],
    )
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"]
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"])
    return train_dataloader, val_dataloader


def build_dataset(
    csv_path: str,
    li_column: str,
    temp: int,
    clip_value: float,
) -> List[Data]:
    """
    Build dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        li_column (str): Column name for the target property.
        temp (int): Temperature to filter data.
        clip_value (float): Minimum value for clipping the target column.

    Returns:
        list: List of PyTorch Geometric Data objects.
    """
    df = pd.read_csv(csv_path)
    df[li_column] = df[li_column].clip(lower=clip_value)
    mpids = df[df["temperature"] == temp]["mpid"].to_list()
    docs = query_mpid_structure(mpids=mpids)

    dataset = []
    for doc in tqdm(docs, desc="Building dataset"):
        material_id = doc["material_id"]
        structure = doc["structure"]

        atoms = AseAtomsAdaptor.get_atoms(structure)
        log_diffusion = np.log10(
            df[(df["mpid"] == material_id) & (df["temperature"] == temp)][
                li_column
            ].iloc[0]
        )

        dataset.append(Data({"atoms": atoms, "log_diffusion": log_diffusion}))

    return dataset


class AtomsToGraphCollater(Collater):
    """
    Collater to convert atomic structures into graph representations.
    """

    def __init__(
        self,
        cutoff: float,
        noise_std: float,
        properties_predictor,
        forces_divided_by_mass: bool,
        shift: bool,
        num_agg: int,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        super().__init__([], follow_batch, exclude_keys)
        self.cutoff = cutoff
        self.noise_std = noise_std
        self.properties_predictor = properties_predictor
        self.forces_divided_by_mass = forces_divided_by_mass
        self.num_agg = num_agg
        self.shift = shift

    def set_noise_to_structures(self, batch: List[Any]) -> Any:
        """
        Add noise to atomic structures.

        Args:
            batch (list): List of atomic structures.

        Returns:
            tuple: Updated batch and applied noises.
        """
        noises = []
        for atoms in batch:
            positions = atoms.get_positions()
            noise = np.random.normal(loc=0, scale=self.noise_std, size=positions.shape)
            atoms.set_positions(positions + noise)
            noises.append(noise)
        return batch, noises

    def set_noise_to_structures_agg(self, batch: List[Any], num_agg: int) -> Any:
        """
        Add noise to atomic structures for aggregation.

        Args:
            batch (list): List of atomic structures.
            num_agg (int): Number of aggregated noisy structures to create.

        Returns:
            tuple: Updated batch and applied noises.
        """
        noises = []
        for atoms in batch:
            for _ in range(num_agg):
                new_atoms = deepcopy(atoms)
                positions = new_atoms.get_positions()
                noise = np.random.normal(
                    loc=0, scale=self.noise_std, size=positions.shape
                )
                new_atoms.set_positions(positions + noise)
            noises.append(new_atoms)
        return batch, noises

    def transit(
        self, mass, atoms_batch, noise_structures_batch, forces_batch, log_diffusion
    ) -> Any:
        """
        Convert noisy structures to graph data.

        Args:
            mass (numpy.ndarray): Masses of atoms.
            noise_structures_batch (list): List of noisy atomic structures.
            forces_batch (list): List of forces for each structure.
            log_diffusion (float): Target property value.

        Returns:
            list: List of graph data objects.
        """
        atoms_list = []
        mass = (
            torch.from_numpy(mass)
            .to(forces_batch[0].device)
            .type(forces_batch[0].dtype)
        )

        for atoms, noise_structures, forces in zip(
            atoms_batch, noise_structures_batch, forces_batch
        ):
            forces_mag = forces.norm(dim=1, keepdim=True)
            factor = torch.log(1.0 + 100.0 * forces_mag) / forces_mag
            value = factor * forces

            edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
                "ijS", a=noise_structures, cutoff=self.cutoff, self_interaction=True
            )

            if self.shift:
                shift = torch.tensor(
                    noise_structures.get_positions() - atoms.get_positions(),
                    dtype=torch.float32,
                    device=value.device,
                )
                value = torch.cat((value, shift), dim=1)

            data = Data(
                pos=torch.tensor(noise_structures.get_positions(), dtype=torch.float32),
                x=value,
                lattice=torch.tensor(
                    noise_structures.cell.array, dtype=torch.float32
                ).unsqueeze(0),
                edge_index=torch.stack(
                    [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
                ),
                edge_shift=torch.tensor(edge_shift, dtype=torch.float32),
                target=torch.tensor(log_diffusion, dtype=torch.float32),
            )

            atoms_list.append(data)

        return atoms_list

    def __call__(self, batch: List[Any]) -> Any:
        """
        Process a batch of atomic data into graph data.

        Args:
            batch (list): List of atomic data objects.

        Returns:
            list: List of processed graph data.
        """
        atoms_batch = [data.x["atoms"] for data in batch]
        noise_structures_batch, _ = self.set_noise_to_structures(deepcopy(atoms_batch))
        data_list = []
        num_atoms = []

        for data in batch:
            num_atoms.append(len(data.x["atoms"]))
            atoms_batch = [data.x["atoms"] for _ in range(self.num_agg)]
            noise_structures_batch, _ = self.set_noise_to_structures_agg(
                deepcopy(atoms_batch), self.num_agg
            )
            forces_batch = self.properties_predictor.predict(noise_structures_batch)[
                "forces"
            ]

            mass = data.x["atoms"].get_masses()
            log_diffusion = data.x["log_diffusion"]

            graphs_batch = self.transit(
                mass, atoms_batch, noise_structures_batch, forces_batch, log_diffusion
            )
            data_list.extend(graphs_batch)
        return super().__call__(data_list), num_atoms
