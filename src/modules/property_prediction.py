from sevenn.train.dataload import graph_build
from torch_geometric.loader import DataLoader
from sevenn.train.dataset import AtomGraphDataset
from ase.calculators.singlepoint import SinglePointCalculator
from sevenn.train.dataload import _set_atoms_y
from typing import Any, List, Optional
import ase.io
import numpy as np
import torch
from torch_geometric.loader.dataloader import Collater
import sevenn


def assign_dummy_y(atoms):
    dummy = {'energy': np.nan, 'free_energy': np.nan}
    dummy['forces'] = np.full((len(atoms), 3), np.nan) 
    dummy['stress'] = np.full((6,), np.nan)  
    calc = SinglePointCalculator(atoms, **dummy)
    atoms = calc.get_atoms()
    return calc.get_atoms()


class SevenNetPropertiesPredictor():
    def __init__(
            self,
            config_name,
        ):

        checkpoint = sevenn.util.pretrained_name_to_path(config_name)
        sevennet_model, sevennet_config = sevenn.util.model_from_checkpoint(checkpoint)

        self.sevennet_model = sevennet_model
        self.sevennet_config = sevennet_config

    def predict(self, batch: List[Any]) -> List[Any]:

        atoms_list = []    
        atoms_len = []
        for atoms in batch:
            atoms_list.append(assign_dummy_y(atoms))
            atoms_len.append(atoms.get_positions().shape[0])

        atoms_list = _set_atoms_y(atoms_list)

        sevennet_data_list = graph_build(
                    atoms_list,
                    self.sevennet_config['cutoff'],
                    num_cores=max(1, self.sevennet_config['_num_workers']),
                    y_from_calc=False,
                )


        sevennet_inference_set = AtomGraphDataset(sevennet_data_list, self.sevennet_config['cutoff'])
        sevennet_inference_set.x_to_one_hot_idx(self.sevennet_config['_type_map'])
        sevennet_inference_set.toggle_requires_grad_of_data(sevenn._keys.POS, True)
        sevennet_infer_list = sevennet_inference_set.to_list()

        sevennet_batch = DataLoader(sevennet_infer_list, batch_size=len(sevennet_infer_list), shuffle=False)

        (sevennet_batch,) = sevennet_batch
        sevennet_output = self.sevennet_model(sevennet_batch).detach().to("cpu")

        forces = []
        energies = []
        total_lenn = 0

        for index, lenn in enumerate(atoms_len):
            forces.append(sevennet_output.inferred_force[total_lenn:total_lenn+lenn, :].clone().detach())
            energies.append(sevennet_output.inferred_total_energy[index].clone().detach())
            total_lenn += lenn
        
        return  {
            'forces': forces,
            'energy': energies,
        }