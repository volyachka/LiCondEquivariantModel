"""
Dataset utilities and data processing for the SevenNet model.
"""

# Standard library imports
import os
import json
from copy import deepcopy
from typing import Any, List, Tuple, Optional, Literal

# Third-party imports
import joblib
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from ase.io import read
from ase.neighborlist import NeighborList
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms
from modules.utils import get_cleaned_neighbours

# First-party imports
from modules.utils import query_mpid_structure


def build_dataloaders_from_dataset(
    dataset: List[Data], test_size: int, random_state: int, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a dataset into training and validation sets, and creates DataLoaders for each.
    """
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        random_state=random_state,
    )

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def build_superionic_toy_dataset(  # pylint: disable=R0914
    root_folder: str, clip_value: float, cutoff: float
) -> List[Data]:
    """
    Constructs a dataset of atomic structures with diffusion properties
    based on superionic toy inputs.
    """
    dataset = []

    run_folder = os.path.join(root_folder, "runs")
    slopes_file = os.path.join(root_folder, "slopes.json")

    with open(slopes_file, "r", encoding="utf-8") as file:
        slopes_info = json.load(file)

    index = 0
    for part in sorted(os.listdir(run_folder)):
        path = os.path.join(run_folder, part)
        for processes in os.listdir(path):
            if processes == ".gitignore":
                continue
            if ".output" not in processes and ".dvc" not in processes:
                relaxed_structure = read(
                    os.path.join(path, processes, "relax_02.traj"), index=-1
                )
                calculator_path = os.path.join(path, processes, "atoms.pkl")
                relaxed_structure.calc = joblib.load(calculator_path).calc

                name = os.path.join(part, processes)
                if name in slopes_info.keys():
                    log_diffusion = {}
                    for element, value in slopes_info[name].items():
                        log_diffusion[element] = np.log10(max(clip_value, value))
                    relaxed_structure.info["id"] = index
                    relaxed_structure.info["name"] = name
                    index += 1
                    cuttofs = len(relaxed_structure.get_positions()) * [cutoff / 2]
                    nl = NeighborList(
                        cuttofs, self_interaction=True, bothways=False, skin=0.5
                    )
                    dataset.append(
                        Data(
                            {
                                "atoms": relaxed_structure,
                                "log_diffusion": log_diffusion,
                                "nl": nl,
                            }
                        )
                    )
    return dataset


def build_dataset(  # pylint: disable=R0914
    csv_path: str, li_column: str, temp: int, clip_value: float, cutoff: float
) -> List[Data]:
    """
    Builds a dataset from a CSV file by processing and filtering data based on specified parameters.
    """
    df = pd.read_csv(csv_path)
    df[li_column] = df[li_column].clip(lower=clip_value)
    mpids = df[df["temperature"] == temp]["mpid"].to_list()
    docs = query_mpid_structure(mpids=mpids)

    dataset = []
    for index, doc in enumerate(docs):
        material_id = doc["material_id"]
        structure = doc["structure"]

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.info["id"] = index
        log_diffusion = np.log10(
            df[(df["mpid"] == material_id) & (df["temperature"] == temp)][
                li_column
            ].iloc[0]
        )

        cuttofs = len(atoms.get_positions()) * [cutoff]
        nl = NeighborList(cuttofs, self_interaction=True, bothways=True, skin=0.5)

        dataset.append(Data({"atoms": atoms, "log_diffusion": log_diffusion, "nl": nl}))

    return dataset


class AtomsToGraphCollater(Collater):  # pylint: disable=R0902
    """
    Collater to convert atomic structures into graph representations.
    """

    def __init__(  # pylint: disable=R0913
        self,
        *,
        dataset: List[Data],
        cutoff: float,
        noise_std: float,
        properties_predictor,
        forces_divided_by_mass: bool,
        use_displacements: bool,
        use_energies: bool,
        num_noisy_configurations: int,
        upd_neigh_style: Literal["update_class", "call_func"],
        predict_per_atom: bool,
        clip_value: float,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        super().__init__([], follow_batch, exclude_keys)
        self.cutoff = cutoff
        self.noise_std = noise_std
        self.properties_predictor = properties_predictor
        self.forces_divided_by_mass = forces_divided_by_mass
        self.num_noisy_configurations = num_noisy_configurations
        self.use_displacements = use_displacements
        self.use_energies = use_energies
        self.upd_neigh_style = upd_neigh_style
        self.predict_per_atom = predict_per_atom
        self.clip_value = clip_value

        assert self.upd_neigh_style in {"update_class", "call_func"}

        self.nl_builders = {}

        if self.upd_neigh_style == "update_class":
            for data in dataset:
                self.nl_builders[data.x["atoms"].info["id"]] = data.x["nl"]

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
            new_atoms = deepcopy(atoms)
            positions = new_atoms.get_positions()
            noise = np.random.normal(loc=0, scale=self.noise_std, size=positions.shape)
            new_atoms.set_positions(positions + noise)
            noises.append(new_atoms)
        return noises

    def transit(  # pylint: disable=R0913, R0914, R0915, E0606
        self,
        *,
        masses_batch: list[np.ndarray],
        atoms_batch: list[MSONAtoms],
        noise_structure_batch: list[MSONAtoms],
        forces_batch: list[torch.Tensor],
        energy_batch: list[torch.Tensor],
        log_diffusion_batch: list[float],
        index_batch: list[int],
    ) -> list[Data]:
        """
        Convert noisy structures to graph data.
        """
        atoms_list = []

        for (
            mass,
            initial_atoms,
            noise_structure,
            forces,
            energy,
            log_diffusion,
            index,
        ) in zip(
            masses_batch,
            atoms_batch,
            noise_structure_batch,
            forces_batch,
            energy_batch,
            log_diffusion_batch,
            index_batch,
            strict=True,
        ):
            mass = (
                torch.from_numpy(mass)
                .to(forces_batch[0].device)
                .type(forces_batch[0].dtype)
            )

            if self.forces_divided_by_mass:
                forces_divided_by_mass = forces / mass[:, None]
                forces_divided_by_mass_mag = forces_divided_by_mass.norm(
                    dim=1, keepdim=True
                )

                assert torch.isnan(forces_divided_by_mass_mag).any().item() is False

                factor = (
                    torch.log(1.0 + 1000.0 * forces_divided_by_mass_mag)
                    / forces_divided_by_mass_mag
                )

                value = factor * forces_divided_by_mass
                value = torch.nan_to_num(value)

            else:
                forces_mag = forces.norm(dim=1, keepdim=True)
                factor = torch.log(1.0 + 100.0 * forces_mag) / forces_mag
                value = factor * forces

                value = torch.nan_to_num(value)

            match self.upd_neigh_style:
                case "update_class":
                    self.nl_builders[index].update(noise_structure)

                    edge_src, edge_dst, edge_shift = get_cleaned_neighbours(
                        self.nl_builders[index], noise_structure, self.cutoff
                    )

                case "call_func":
                    edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
                        "ijS",
                        a=noise_structure,
                        cutoff=self.cutoff,
                        self_interaction=True,
                    )

                case _:
                    raise NotImplementedError(self.upd_neigh_style)

            if self.use_displacements:
                shift = torch.tensor(
                    noise_structure.get_positions() - initial_atoms.get_positions(),
                    dtype=torch.float32,
                    device=value.device,
                )
                value = torch.cat((value, shift), dim=1)

            if self.use_energies:
                value = torch.cat((value, energy.repeat(value.shape[0], 1)), dim=1)

            if self.predict_per_atom:
                target = torch.full(
                    (forces.shape[0],), self.clip_value, device=value.device
                )
                symbols = np.array(noise_structure.get_chemical_symbols())

                if isinstance(log_diffusion, dict):
                    for element, diffusion in log_diffusion.items():
                        mask = symbols == element
                        target[mask] = diffusion
                else:
                    mask = symbols == "Li"
                    target[mask] = log_diffusion
            else:
                target = torch.tensor(
                    log_diffusion, dtype=torch.float32, device=value.device
                )

            assert torch.isnan(value).any().item() is False

            data = Data(
                pos=torch.tensor(
                    noise_structure.get_positions(),
                    dtype=torch.float32,
                    device=value.device,
                ),
                x=value,
                y=target,
                lattice=torch.tensor(
                    noise_structure.cell.array, dtype=torch.float32
                ).unsqueeze(0),
                edge_index=torch.stack(
                    [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
                ),
                edge_shift=torch.tensor(edge_shift, dtype=torch.float32),
                symbols=np.array(noise_structure.get_chemical_symbols()),
                index=index,
            )

            atoms_list.append(data)
        return atoms_list

    def __call__(self, batch: List[Any]) -> Any:  # pylint: disable=R0914
        """
        Process a batch of atomic data into graph data.

        Args:
            batch (list): List of atomic data objects.

        Returns:
            list: List of processed graph data.
        """

        masses_batch = []
        log_diffusion_batch = []
        atoms_batch = []
        index_batch = []
        num_atoms = []

        atoms_reference_positions = np.concatenate(
            [data.x["atoms"].get_positions() for data in batch]
        )

        for data in batch:
            masses_batch.extend(
                self.num_noisy_configurations * [data.x["atoms"].get_masses()]
            )
            log_diffusion_batch.extend(
                self.num_noisy_configurations * [data.x["log_diffusion"]]
            )
            atoms_batch.extend(self.num_noisy_configurations * [data.x["atoms"]])
            num_atoms.append(len(data.x["atoms"]))
            index_batch.extend(
                self.num_noisy_configurations * [data.x["atoms"].info["id"]]
            )

        noise_structure_batch = self.set_noise_to_structures(atoms_batch)
        with torch.enable_grad():
            properites = self.properties_predictor.predict(noise_structure_batch)

        forces_batch = properites["forces"]
        energy_batch = properites["energy"]

        if self.use_energies:
            equilibrium_structures_batch = list(data.x["atoms"] for data in batch)
            energy_equilibrium_batch = self.properties_predictor.predict(
                equilibrium_structures_batch
            )["energy"]

            energy_equilibrium_batch = torch.Tensor(
                energy_equilibrium_batch
            ).repeat_interleave(self.num_noisy_configurations)

            energy_batch = [
                energy - energy_equilibrium
                for energy, energy_equilibrium in zip(
                    energy_batch, energy_equilibrium_batch, strict=True
                )
            ]
        graphs_batch = self.transit(
            masses_batch=masses_batch,
            atoms_batch=atoms_batch,
            noise_structure_batch=noise_structure_batch,
            forces_batch=forces_batch,
            energy_batch=energy_batch,
            log_diffusion_batch=log_diffusion_batch,
            index_batch=index_batch,
        )

        atoms_positions_should_be_unchanged = np.concatenate(
            [data.x["atoms"].get_positions() for data in batch]
        )
        assert (atoms_reference_positions == atoms_positions_should_be_unchanged).all()

        return super().__call__(graphs_batch), num_atoms
