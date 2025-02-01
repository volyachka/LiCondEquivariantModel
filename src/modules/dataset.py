"""
Dataset utilities and data processing for the SevenNet model.
"""

# Standard imports
# Standard library imports
from copy import deepcopy
from typing import Any, List, Tuple, Optional

# Third-party imports
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from ase.neighborlist import NeighborList
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms


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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


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
        nl = NeighborList(cuttofs, self_interaction=False, bothways=True, skin=0.5)

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
        upd_neigh_style: str,
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

    def transit(  # pylint: disable=R0913, R0914, E0606
        self,
        *,
        masses_batch: list[np.ndarray],
        atoms_batch: list[MSONAtoms],
        noise_structures_batch: list[MSONAtoms],
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
            noise_structures,
            forces,
            energy,
            log_diffusion,
            index,
        ) in zip(
            masses_batch,
            atoms_batch,
            noise_structures_batch,
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
                factor = (
                    torch.log(1.0 + 1000.0 * forces_divided_by_mass_mag)
                    / forces_divided_by_mass_mag
                )
                value = factor * forces_divided_by_mass
            else:
                forces_mag = forces.norm(dim=1, keepdim=True)
                factor = torch.log(1.0 + 100.0 * forces_mag) / forces_mag
                value = factor * forces
                assert torch.isnan(value).any().item() is False

            pos = noise_structures.get_positions()
            lattice = noise_structures.cell.array

            if self.upd_neigh_style == "update_class":

                self.nl_builders[index].update(noise_structures)

                src = []
                dst = []
                vec = []

            for i_atom, _ in enumerate(initial_atoms.symbols):
                neighbor_ids, offsets = self.nl_builders[index].get_neighbors(
                    i_atom
                )
                rr = pos[neighbor_ids] + offsets @ lattice - pos[i_atom][None, :]
                suitable_neigh = np.linalg.norm(rr, axis=1) <= self.cutoff
                neighbor_ids = neighbor_ids[suitable_neigh]
                offsets = offsets[suitable_neigh]
                rr = rr[suitable_neigh]

                src.append(np.ones(len(neighbor_ids)) * i_atom)
                dst.append(neighbor_ids)
                vec.append(rr)

            edge_vec = np.vstack(vec)
            edge_dst = np.concatenate(dst)
            edge_src = np.concatenate(src)

            if self.upd_neigh_style == "call_func":
                edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
                    "ijS", a=noise_structures, cutoff=self.cutoff, self_interaction=True
                )

                edge_vec = pos[edge_dst] - pos[edge_src] + (edge_shift @ lattice)

            if self.use_displacements:
                shift = torch.tensor(
                    noise_structures.get_positions() - initial_atoms.get_positions(),
                    dtype=torch.float32,
                    device=value.device,
                )
                value = torch.cat((value, shift), dim=1)

            if self.use_energies:
                value = torch.cat((value, energy.repeat(value.shape[0], 1)), dim=1)

            edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list(
                "ijS", a=noise_structures, cutoff=self.cutoff, self_interaction=True
            )

            # data = Data(
            #     pos=torch.tensor(noise_structures.get_positions(), dtype=torch.float32),
            #     x=value,
            #     lattice=torch.tensor(
            #         noise_structures.cell.array, dtype=torch.float32
            #     ).unsqueeze(0),
            #     edge_index=torch.stack(
            #         [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
            #     ),
            #     edge_shift=torch.tensor(edge_shift, dtype=torch.float32),
            #     target=torch.tensor(log_diffusion, dtype=torch.float32),
            # )

            data = Data(
                pos=torch.tensor(
                    noise_structures.get_positions(),
                    dtype=torch.float32,
                    device=value.device,
                ),
                x=value,
                edge_src=torch.LongTensor(deepcopy(edge_src)),
                edge_dst=torch.LongTensor(deepcopy(edge_dst)),
                edge_vec=torch.tensor(
                    edge_vec, dtype=torch.float32, device=value.device
                ),
                target=torch.tensor(
                    log_diffusion, dtype=torch.float32, device=value.device
                ),
                lattice=torch.tensor(
                    noise_structures.cell.array, dtype=torch.float32
                ).unsqueeze(0),
                edge_index=torch.stack(
                    [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
                ),
                edge_shift=torch.tensor(edge_shift, dtype=torch.float32),
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

        noise_structures_batch = self.set_noise_to_structures(atoms_batch)

        with torch.enable_grad():
            properites = self.properties_predictor.predict(noise_structures_batch)
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
            noise_structures_batch=noise_structures_batch,
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
