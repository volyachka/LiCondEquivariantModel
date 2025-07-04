"""
This module defines the `SevenNetPropertiesPredictor` class, which uses a pretrained
SevenNet model to predict properties (such as forces and energy) of atomic systems.
It also defines the `LennardJonesPropertiesPredictor` class, which uses the
Lennard-Jones potential for property prediction.
"""

# Standard imports
from typing import Any, List
from copy import deepcopy

# Third-party imports
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator

# Local imports
import sevenn
from sevenn.train.dataload import graph_build, _set_atoms_y
from sevenn.train.dataset import AtomGraphDataset


def assign_dummy_y(atoms):
    """
    Assign dummy property values (energy, forces, stress) to the atoms.

    Args:
        atoms (ase.Atoms): The atomic structure for which the properties are assigned.

    Returns:
        ase.Atoms: The updated atomic structure with dummy properties.
    """
    dummy = {"energy": np.nan, "free_energy": np.nan}
    dummy["forces"] = np.full((len(atoms), 3), np.nan)
    dummy["stress"] = np.full((6,), np.nan)
    calc = SinglePointCalculator(atoms, **dummy)
    atoms = calc.get_atoms()
    return calc.get_atoms()


class SevenNetPropertiesPredictor:  # pylint: disable=R0903
    """
    A class that uses a pretrained SevenNet model to predict atomic properties such as forces
    and energy.

    Attributes:
        device (str): The device on which the model should run (e.g., "cuda" or "cpu").
        sevennet_model (torch.nn.Module): The pretrained SevenNet model for property prediction.
        sevennet_config (dict): Configuration dictionary for the SevenNet model.

    Methods:
        predict: Makes predictions for forces and energy based on a batch of atomic structures.
    """

    def __init__(self, device, predictor_config) -> None:
        """
        Initializes the `SevenNetPropertiesPredictor` with the specified configuration name
        and device.

        Args:
            config_name (str): The name of the pretrained model configuration.
            device (str): The device to run the model on (e.g., "cpu" or "cuda").
        """
        checkpoint = sevenn.util.pretrained_name_to_path(predictor_config["checkpoint"])
        sevennet_model, sevennet_config = sevenn.util.model_from_checkpoint(checkpoint)

        self.batch_size = predictor_config["batch_size"]
        self.device = device
        self.sevennet_model = sevennet_model
        self.sevennet_config = sevennet_config
        self.sevennet_model = self.sevennet_model.to(self.device)

    def predict(self, batch: List[Any]) -> dict:
        """
        Predicts the forces and energy for a batch of atomic structures using the pretrained
        SevenNet model.

        Args:
            batch (List[ase.Atoms]): A list of atomic structures to make predictions for.

        Returns:
            dict: A dictionary containing predicted forces and energies.
                - "forces": List of predicted forces for each structure.
                - "energy": List of predicted energies for each structure.
        """
        atoms_list = []
        num_atoms_per_structure = []
        for atoms in batch:
            atoms_list.append(assign_dummy_y(atoms))
            num_atoms_per_structure.append(atoms.get_positions().shape[0])

        atoms_list = _set_atoms_y(atoms_list)

        sevennet_data_list = graph_build(
            atoms_list,
            self.sevennet_config["cutoff"],
            num_cores=max(1, self.sevennet_config["_num_workers"]),
            y_from_calc=False,
        )

        sevennet_inference_set = AtomGraphDataset(
            sevennet_data_list, self.sevennet_config["cutoff"]
        )
        sevennet_inference_set.x_to_one_hot_idx(self.sevennet_config["_type_map"])
        # pylint: disable=W0212
        sevennet_inference_set.toggle_requires_grad_of_data(sevenn._keys.POS, True)
        sevennet_infer_list = sevennet_inference_set.to_list()

        sevennet_data = DataLoader(
            sevennet_infer_list, batch_size=self.batch_size, shuffle=False
        )

        forces = []
        energies = []

        with torch.enable_grad():
            for sevennet_batch in sevennet_data:
                sevennet_batch = sevennet_batch.to(self.device)
                sevennet_output = self.sevennet_model(sevennet_batch)
                forces.append(sevennet_output.inferred_force.detach())
                energies.append(sevennet_output.inferred_total_energy.detach())

        energies = list(torch.split(torch.cat(energies), 1))
        forces = list(torch.split(torch.cat(forces), num_atoms_per_structure, dim=0))
        assert len(forces) == len(energies) == len(num_atoms_per_structure)
        assert isinstance(forces, list) and isinstance(energies, list)

        return {
            "forces": forces,
            "energy": energies,
        }


class ZeroValuesPropertiesPredictor:  # pylint: disable=R0903
    """
    A class that predicts atomic properties, such as forces and energy,
    using random predicitons.
    """

    def __init__(self, device) -> None:
        self.device = device

    def predict(self, batch: List[Any]) -> dict:
        """
        Predicts atomic forces and energies for a batch of atomic structures.
        """
        forces = []
        energies = []

        for atoms in batch:
            forces_atoms = torch.zeros(atoms.get_positions().shape, device=self.device)
            energy_atoms = torch.zeros((1), device=self.device)
            forces.append(forces_atoms)
            energies.append(energy_atoms)

        assert len(forces) == len(energies)
        assert isinstance(forces, list) and isinstance(energies, list)
        return {
            "forces": forces,
            "energy": energies,
        }

class RandomPropertiesPredictor:  # pylint: disable=R0903
    """
    A class that predicts atomic properties, such as forces and energy,
    using random predicitons.
    """

    def __init__(self, device) -> None:
        self.device = device

    def predict(self, batch: List[Any]) -> dict:
        """
        Predicts atomic forces and energies for a batch of atomic structures.
        """
        forces = []
        energies = []

        for atoms in batch:
            forces_atoms = torch.normal(
                mean=0, std=torch.ones(atoms.get_positions().shape, device=self.device)
            )
            energy_atoms = torch.normal(
                mean=0, std=torch.ones((1), device=self.device)
            ).to(self.device)
            forces.append(forces_atoms)
            energies.append(energy_atoms)

        assert len(forces) == len(energies)
        assert isinstance(forces, list) and isinstance(energies, list)
        return {
            "forces": forces,
            "energy": energies,
        }


class MDTrajectoryPropertiesPredictor:

    def __init__(self, device) -> None:
        self.device = device

    def predict(self, batch: List[Any]) -> dict:
        forces = []
        energies = []
        
        for atoms in batch:
            forces_atoms =  torch.tensor(
                atoms.get_forces(),
                device=self.device,
                dtype=torch.float32,
                )
            energy_atoms =  torch.tensor(
                                atoms.get_potential_energy(),
                                device=self.device,
                                dtype=torch.float32,
                        ).unsqueeze(0)
            
            forces.append(forces_atoms)
            energies.append(energy_atoms)

        assert len(forces) == len(energies)
        assert isinstance(forces, list) and isinstance(energies, list)
        return {
            "forces": forces,
            "energy": energies,
        }
    
class AseCalculatorPropertiesPredictor:  # pylint: disable=R0903
    """
    A class that predicts atomic properties, such as forces and energy,
    using the Lennard-Jones potential.
    """

    def __init__(self, device, name, predictor_config, dataset) -> None:
        self.device = device
        self.reference_atoms = {}
        for data in dataset:
            atoms = deepcopy(data.x["atoms"])
            if name.lower() == "lennardjones":
                atoms.calc = LennardJones(**predictor_config)
            assert atoms.calc is not None
            self.reference_atoms[atoms.info["id"]] = atoms

    def predict(self, batch: List[Any]) -> dict:
        """
        Predicts atomic forces and energies for a batch of atomic structures.
        """
        forces = []
        energies = []
        num_atoms_per_structure = []

        for atoms in batch:
            reference = self.reference_atoms[atoms.info["id"]]
            reference.set_positions(atoms.positions)
            num_atoms_per_structure.append(atoms.get_positions().shape[0])
            energy_atoms = torch.tensor(
                reference.get_potential_energy(),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            forces_atoms = torch.tensor(
                reference.get_forces(), device=self.device, dtype=torch.float32
            )
            if torch.isnan(forces_atoms).any() is False:
                forces_atoms = torch.nan_to_num(forces_atoms)

            forces.append(forces_atoms)
            energies.append(energy_atoms)

        assert len(forces) == len(energies) == len(num_atoms_per_structure)
        assert isinstance(forces, list) and isinstance(energies, list)
        return {
            "forces": forces,
            "energy": energies,
        }


# class SevenNetEmbeddingPredictor:  # pylint: disable=R0903
#     """
#     A class that uses a pretrained SevenNet model to predict atomic properties such as forces
#     and energy.

#     Attributes:
#         device (str): The device on which the model should run (e.g., "cuda" or "cpu").
#         sevennet_model (torch.nn.Module): The pretrained SevenNet model for property prediction.
#         sevennet_config (dict): Configuration dictionary for the SevenNet model.

#     Methods:
#         predict: Makes predictions for forces and energy based on a batch of atomic structures.
#     """

#     def __init__(self, device, predictor_config) -> None:
#         """
#         Initializes the `SevenNetPropertiesPredictor` with the specified configuration name
#         and device.

#         Args:
#             config_name (str): The name of the pretrained model configuration.
#             device (str): The device to run the model on (e.g., "cpu" or "cuda").
#         """
#         checkpoint = sevenn.util.pretrained_name_to_path(predictor_config["checkpoint"])
#         sevennet_model, sevennet_config = sevenn.util.model_from_checkpoint(checkpoint)

#         self.batch_size = predictor_config["batch_size"]
#         self.device = device
#         self.sevennet_model = sevennet_model
#         self.sevennet_config = sevennet_config
#         self.sevennet_model = self.sevennet_model.to(self.device)

#     def predict(self, batch: List[Any]) -> dict:
#         """
#         Predicts the forces and energy for a batch of atomic structures using the pretrained
#         SevenNet model.

#         Args:
#             batch (List[ase.Atoms]): A list of atomic structures to make predictions for.

#         Returns:
#             dict: A dictionary containing predicted forces and energies.
#                 - "forces": List of predicted forces for each structure.
#                 - "energy": List of predicted energies for each structure.
#         """
#         atoms_list = []
#         num_atoms_per_structure = []
#         for atoms in batch:
#             atoms_list.append(assign_dummy_y(atoms))
#             num_atoms_per_structure.append(atoms.get_positions().shape[0])

#         atoms_list = _set_atoms_y(atoms_list)

#         sevennet_data_list = graph_build(
#             atoms_list,
#             self.sevennet_config["cutoff"],
#             num_cores=max(1, self.sevennet_config["_num_workers"]),
#             y_from_calc=False,
#         )

#         sevennet_inference_set = AtomGraphDataset(
#             sevennet_data_list, self.sevennet_config["cutoff"]
#         )
#         sevennet_inference_set.x_to_one_hot_idx(self.sevennet_config["_type_map"])
#         # pylint: disable=W0212
#         sevennet_inference_set.toggle_requires_grad_of_data(sevenn._keys.POS, True)
#         sevennet_infer_list = sevennet_inference_set.to_list()

#         sevennet_data = DataLoader(
#             sevennet_infer_list, batch_size=self.batch_size, shuffle=False
#         )

#         for sevennet_batch in sevennet_data:
#             sevennet_batch = sevennet_batch.to(self.device)
#             sevennet_output = self.sevennet_model(sevennet_batch)
#             forces.append(sevennet_output.inferred_force.detach())
#             energies.append(sevennet_output.inferred_total_energy.detach())

#         energies = list(torch.split(torch.cat(energies), 1))
#         forces = list(torch.split(torch.cat(forces), num_atoms_per_structure, dim=0))
#         assert len(forces) == len(energies) == len(num_atoms_per_structure)
#         assert isinstance(forces, list) and isinstance(energies, list)

#         return {
#             "forces": forces,
#             "energy": energies,
#         }
