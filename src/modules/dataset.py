"""
Dataset utilities and data processing for the SevenNet model.
"""

# Standard library imports
import os
import json
from copy import deepcopy
from typing import Any, List, Tuple, Optional, Literal
import random

# Third-party imports
import numpy as np
import pandas as pd
import torch
import joblib
from joblib import Memory
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater

# ASE imports (grouped)
import ase
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList
from ase.calculators.singlepoint import SinglePointCalculator

# Pymatgen imports
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms

# First-party imports
from modules.utils import (
    get_cleaned_neighbours,
    get_cleaned_neighbours_li_grouped,
    query_mpid_structure,
)
from modules.property_prediction import SevenNetPropertiesPredictor


def split_dataset_train_val_part(
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


_memory = Memory("/mnt/hdd/turchina/extended_sevennet")


@_memory.cache
def _load_traj_cached(traj_path: str, skip_first_fs: int, step_size_fs: int):
    assert step_size_fs is not None
    structs = read(traj_path, index=slice(skip_first_fs, None, step_size_fs))
    reference_symbols = list(map(str, structs[0].get_chemical_symbols()))
    reference_cell = structs[0].cell.array

    for s in structs:
        if list(map(str, s.get_chemical_symbols())) != reference_symbols:
            raise ValueError("All structures must have the same atomic symbols.")
        if not np.allclose(s.cell.array, reference_cell):
            raise ValueError("All structures must have the same cell dimensions.")

    symbols = list(map(str, structs[0].symbols))
    cell = structs[0].cell.array
    xyz = np.stack([s.positions for s in structs])
    energies = np.stack([s.get_potential_energy() for s in structs])
    forces = np.stack([s.get_forces() for s in structs])

    return {
        "symbols": symbols,
        "cell": cell,
        "xyz": xyz,
        "energies": energies,
        "forces": forces,
    }


def _from_dict_to_snapshots(data: dict):
    snapshots = []
    for pos, forces, energy in zip(data["xyz"], data["forces"], data["energies"]):
        atoms = Atoms(symbols=data["symbols"], positions=pos, cell=data["cell"])
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        snapshots.append(atoms)
    return snapshots


def build_extended_sevennet(  # pylint: disable=R0913, R0914, R0917, R1702
    root_folders: List[str],
    clip_value: float,
    cutoff: float,
    strategy_sampling: str,
    skip_first_fs: Any = None,
    step_size_fs: Any = None,
):
    """
    Constructs an extended dataset of atomic structures with diffusion properties
    based on sevennet inputs.
    """
    assert strategy_sampling in {
        "trajectory",
        "gaussian_noise",
        "trajectory_per_interval",
    }
    dataset = []
    idx = 0

    for root_folder in root_folders:
        slopes_file = os.path.join(root_folder, "slopes.json")

        with open(slopes_file, "r", encoding="utf-8") as file:
            slopes_info = json.load(file)

        for folder in os.listdir(root_folder):
            if "runs-1000K" in folder:
                for mp_id in os.listdir(os.path.join(root_folder, folder)):
                    if (
                        ".output" not in mp_id
                        and ".dvc" not in mp_id
                        and ".gitignore" not in mp_id
                    ):
                        name = os.path.join(folder, mp_id)
                        if name in slopes_info.keys():
                            log_diffusion = {}
                            for element, value in slopes_info[name].items():
                                log_diffusion[element] = np.log10(
                                    max(clip_value, value)
                                )
                        else:
                            continue

                        full_path = os.path.join(root_folder, folder, mp_id)

                        if strategy_sampling in {
                            "trajectory",
                            "trajectory_per_interval",
                        }:

                            traj_path = os.path.join(full_path, "md.traj")

                            cached_dict_info = _load_traj_cached(
                                traj_path, skip_first_fs, step_size_fs
                            )
                            snapshots = _from_dict_to_snapshots(cached_dict_info)

                            cuttofs = len(snapshots[0].get_positions()) * [cutoff / 2]

                            nl = NeighborList(
                                cuttofs, self_interaction=True, bothways=True, skin=0.5
                            )

                            print(len(snapshots))
                            dataset.append(
                                Data(
                                    {
                                        "idx": idx,
                                        "log_diffusion": log_diffusion,
                                        "snapshots": snapshots,
                                        "nl": nl,
                                    }
                                )
                            )

                        elif strategy_sampling == "gaussian_noise":
                            relaxed_structure_path = os.path.join(
                                full_path, "relax_02.traj"
                            )
                            relaxed_structure = read(relaxed_structure_path, index=-1)
                            calculator_path = os.path.join(full_path, "atoms.pkl")
                            relaxed_structure.calc = joblib.load(calculator_path).calc
                            cuttofs = len(relaxed_structure.get_positions()) * [
                                cutoff / 2
                            ]

                            nl = NeighborList(
                                cuttofs, self_interaction=True, bothways=True, skin=0.5
                            )
                            dataset.append(
                                Data(
                                    {
                                        "atoms": relaxed_structure,
                                        "log_diffusion": log_diffusion,
                                        "nl": nl,
                                        "idx": idx,
                                    }
                                )
                            )

                        print(idx)
                        idx += 1
    return dataset


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

    idx = 0
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
                    relaxed_structure.info["idx"] = idx
                    relaxed_structure.info["name"] = name
                    idx += 1
                    cuttofs = len(relaxed_structure.get_positions()) * [cutoff / 2]
                    nl = NeighborList(
                        cuttofs, self_interaction=True, bothways=True, skin=0.5
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


def build_datasets_with_selected_by_random_samples(  # pylint: disable=R0914
    csv_path: str, li_column: str, temp: int, clip_value: float, cutoff: float
) -> List[Data]:
    """
    Builds a dataset from a CSV file by processing and filtering data based on specified parameters.
    """
    df = pd.read_csv(csv_path)
    df[li_column] = df[li_column].clip(lower=clip_value)
    mpids = df[df["temperature"] == temp]["mpid"].to_list()
    docs = query_mpid_structure(mpids=mpids)

    random_labels = pd.read_csv("data/mpids_random30.txt").iloc[:, 0].tolist()
    sevennet_labels = pd.read_csv("data/mpids_sorted.txt").iloc[:100, 0].tolist()
    random_labels = set(random_labels) - set(sevennet_labels)

    dataset_main = []
    dataset_random = []
    for idx, doc in enumerate(docs):
        material_id = doc["material_id"]
        structure = doc["structure"]

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.info["idx"] = idx

        log_diffusion = np.log10(
            df[(df["mpid"] == material_id) & (df["temperature"] == temp)][
                li_column
            ].iloc[0]
        )

        cuttofs = len(atoms.get_positions()) * [cutoff]
        nl = NeighborList(cuttofs, self_interaction=True, bothways=True, skin=0.5)

        if material_id not in random_labels:
            dataset_main.append(
                Data({"atoms": atoms, "log_diffusion": log_diffusion, "nl": nl})
            )
        else:
            dataset_random.append(
                Data({"atoms": atoms, "log_diffusion": log_diffusion, "nl": nl})
            )
    return dataset_main, dataset_random


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
    for idx, doc in enumerate(docs):
        material_id = doc["material_id"]
        structure = doc["structure"]

        atoms = AseAtomsAdaptor.get_atoms(structure)
        log_diffusion = np.log10(
            df[(df["mpid"] == material_id) & (df["temperature"] == temp)][
                li_column
            ].iloc[0]
        )

        cuttofs = len(atoms.get_positions()) * [cutoff]
        nl = NeighborList(cuttofs, self_interaction=True, bothways=True, skin=0.5)

        dataset.append(
            Data({"atoms": atoms, "log_diffusion": log_diffusion, "nl": nl, "idx": idx})
        )

    return dataset


class AtomsToGraphCollater(Collater):  # pylint: disable=R0902, R0914
    """
    Collater to convert atomic structures into graph representations.
    """

    def __init__(  # pylint: disable=R0913, R0914
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
        clip_value: float,
        strategy_sampling: str,
        node_style_build: str,
        device: Any,
        sample_first_five_seconds: bool = False,
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
        self.clip_value = clip_value
        self.device = device
        self.strategy_sampling = strategy_sampling
        self.node_style_build = node_style_build
        self.sample_first_five_seconds = sample_first_five_seconds

        assert self.upd_neigh_style in {"update_class", "call_func"}
        assert self.strategy_sampling in {
            "gaussian_noise",
            "trajectory",
            "trajectory_per_interval",
        }
        assert self.node_style_build in {"li_grouped", "full_atoms"}
        self.nl_builders = {}

        if self.upd_neigh_style == "update_class":

            for data in dataset:
                self.nl_builders[data.x["idx"]] = data.x["nl"]
            # match self.strategy_sampling:
            #     case "gaussian_noise":
            #         for data in dataset:
            #             self.nl_builders[data.x["atoms"].info["idx"]] = data.x["nl"]
            #     case "trajectory":
            #         for data in dataset:
            #             self.nl_builders[data.x["idx"]] = data.x["nl"]
            #     case _:
            #         raise NotImplementedError(self.strategy_sampling)

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
        change_structures_batch: list[MSONAtoms],
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
            idx,
        ) in zip(
            masses_batch,
            atoms_batch,
            change_structures_batch,
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
                    self.nl_builders[idx].update(noise_structure)

                    if self.node_style_build == "li_grouped":
                        edge_src, edge_dst, edge_shift, _ = (
                            get_cleaned_neighbours_li_grouped(
                                self.nl_builders[idx], noise_structure, self.cutoff
                            )
                        )
                    else:
                        edge_src, edge_dst, edge_shift = get_cleaned_neighbours(
                            self.nl_builders[idx], noise_structure, self.cutoff
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

            target = torch.full(
                (forces.shape[0],), self.clip_value, device=value.device
            )
            idx = torch.full((forces.shape[0],), idx, device=value.device)
            symbols = np.array(noise_structure.get_chemical_symbols())

            if isinstance(log_diffusion, dict):
                unique_symbols = list(log_diffusion.keys())
                for element, diffusion in log_diffusion.items():
                    mask = symbols == element
                    target[mask] = diffusion
            else:
                mask = symbols == "Li"
                target[mask] = log_diffusion

            # else:
            #     target = torch.tensor(
            #         log_diffusion, dtype=torch.float32, device=value.device
            #     )
            # idx = torch.tensor(deepcopy(idx))

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
                idx=idx,
                unique_symbols=np.array(unique_symbols),
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

        match self.strategy_sampling:
            case "gaussian_noise":
                for data in batch:
                    atoms_batch.extend(
                        self.num_noisy_configurations * [data.x["atoms"]]
                    )

                    masses_batch.extend(
                        self.num_noisy_configurations * [data.x["atoms"].get_masses()]
                    )
                    log_diffusion_batch.extend(
                        self.num_noisy_configurations * [data.x["log_diffusion"]]
                    )
                    num_atoms.append(len(data.x["atoms"]))
                    index_batch.extend(self.num_noisy_configurations * [data.x["idx"]])

                change_structures_batch = self.set_noise_to_structures(atoms_batch)

                with torch.enable_grad():
                    properites = self.properties_predictor.predict(
                        change_structures_batch
                    )
                    forces_batch = properites["forces"]
                    energy_batch = properites["energy"]

            case "trajectory" | "trajectory_per_interval":
                for data in batch:
                    if self.strategy_sampling == "trajectory":
                        atoms_batch.extend(
                            random.sample(
                                data.x["snapshots"], self.num_noisy_configurations
                            )
                        )
                    elif self.strategy_sampling == "trajectory_per_interval":

                        if self.sample_first_five_seconds:
                            start_interval = 0
                            end_interval = 50
                            atoms_batch.extend(
                                random.sample(
                                    data.x["snapshots"][start_interval:end_interval],
                                    self.num_noisy_configurations,
                                )
                            )
                        else:
                            start_interval = random.randint(
                                0, len(data.x["snapshots"]) - 50
                            )
                            end_interval = start_interval + 50

                            atoms_batch.extend(
                                random.sample(
                                    data.x["snapshots"][start_interval:end_interval],
                                    self.num_noisy_configurations,
                                )
                            )

                    masses_batch.extend(
                        self.num_noisy_configurations * [atoms_batch[-1].get_masses()]
                    )

                    log_diffusion_batch.extend(
                        self.num_noisy_configurations * [data.x["log_diffusion"]]
                    )

                    num_atoms.append(len(atoms_batch[-1]))
                    index_batch.extend(self.num_noisy_configurations * [data.x["idx"]])

                change_structures_batch = atoms_batch

                forces_batch = []
                energy_batch = []

                if isinstance(self.properties_predictor, SevenNetPropertiesPredictor):
                    for structure in atoms_batch:
                        forces_batch.append(
                            torch.tensor(
                                structure.get_forces(),
                                device=self.device,
                                dtype=torch.float32,
                            )
                        )

                        energy_batch.append(
                            torch.tensor(
                                structure.get_potential_energy(),
                                device=self.device,
                                dtype=torch.float32,
                            ).unsqueeze(0)
                        )

                else:
                    with torch.enable_grad():
                        properites = self.properties_predictor.predict(
                            change_structures_batch
                        )
                        forces_batch = properites["forces"]
                        energy_batch = properites["energy"]

            case _:
                raise NotImplementedError(self.strategy_sampling)

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
            change_structures_batch=change_structures_batch,
            forces_batch=forces_batch,
            energy_batch=energy_batch,
            log_diffusion_batch=log_diffusion_batch,
            index_batch=index_batch,
        )

        # atoms_positions_should_be_unchanged = np.concatenate(
        #     [data.x["atoms"].get_positions() for data in batch]
        # )
        # assert (atoms_reference_positions == atoms_positions_should_be_unchanged).all()

        return super().__call__(graphs_batch), num_atoms


def build_dataset_snapshots_by_sevennet(  # pylint: disable=R0913, R0914, R0917
    csv_path: str,
    li_column: str,
    temp: int,
    clip_value: float,
    cutoff: int,
    skip_first_fs: int = 10000,
    step_size_fs: int = 100,
) -> List[Data]:
    """
    Builds a dataset from a CSV file by processing and filtering data based on specified parameters.
    """
    df = pd.read_csv(csv_path)
    df[li_column] = df[li_column].clip(lower=clip_value)
    mpids = df[df["temperature"] == temp]["mpid"].to_list()
    docs = query_mpid_structure(mpids=mpids)

    dataset = []
    for idx, doc in enumerate(tqdm(docs)):
        print(idx)
        material_id = doc["material_id"]

        log_diffusion = np.log10(
            df[(df["mpid"] == material_id) & (df["temperature"] == temp)][
                li_column
            ].iloc[0]
        )

        traj_path = (
            f"/mnt/hdd/maevskiy/MLIAP-MD-data/gpu_prod/{material_id}-T1000/md.traj"
        )

        snapshots = read(traj_path, index=slice(skip_first_fs, None, step_size_fs))

        cuttofs = len(snapshots[0].get_positions()) * [cutoff / 2]
        nl = NeighborList(cuttofs, self_interaction=True, bothways=True, skin=0.5)

        dataset.append(
            Data(
                {
                    "idx": idx,
                    "log_diffusion": log_diffusion,
                    "snapshots": snapshots,
                    "nl": nl,
                }
            )
        )

    return dataset
