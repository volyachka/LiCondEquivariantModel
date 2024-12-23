from copy import deepcopy
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.loader import DataLoader

from .utils import query_mpid_structure

def build_dataloader_cv(config):

    dataset = build_dataset(config['data']['data_path'], config['data']['target_column'], config['data']['temperature'],  config['data']['clip_value'])
    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=config['data']['test_size'], random_state = config['data']['random_state']) 
  
    train_dataset = [dataset[i] for i in train_indices] 
    val_dataset = [dataset[i] for i in val_indices] 

    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['batch_size'])
    return train_dataloader, val_dataloader

def build_dataset(csv_path = 'data/sevennet_slopes.csv', li_column = 'v1_Li_slope', temp = 1000, clip_value = 1e-4):
    df = pd.read_csv(csv_path)
    df[li_column] = df[li_column].clip(lower=clip_value)
    mpids = df[df['temperature'] == temp]['mpid'].to_list()
    docs = query_mpid_structure(mpids=mpids)

    dataset = []
    num_samples = 0
    for doc in tqdm(docs):
        material_id = doc["material_id"]
        structure = doc["structure"]

        atoms = AseAtomsAdaptor.get_atoms(structure)
        log_diffusion = np.log10(df[(df['mpid'] == material_id) & (df['temperature'] == temp)]['v1_Li_slope'].iloc[0])

        dataset.append(Data({'atoms': atoms,
                            'log_diffusion': log_diffusion
                            }))
    
    return dataset

class AtomsToGraphCollater(Collater):

    def __init__(
        self,
        cutoff: float,
        noise_std: float,
        properties_predictor,
        forces_divided_by_mass,
        num_agg,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        super().__init__([], follow_batch, exclude_keys)
        self.cutoff = cutoff
        self.noise_std = noise_std
        self.properties_predictor = properties_predictor
        self.forces_divided_by_mass = forces_divided_by_mass
        self.num_agg = num_agg

    def set_noise_to_structures(self, batch: List[Any]) -> Any:
        noises = []
        for atoms in batch:
            positions = atoms.get_positions() 
            noise = np.random.normal(loc=0, scale=self.noise_std, size=positions.shape)    
            atoms.set_positions(positions + noise)
            noises.append(noise)
        return batch, noises

    def set_noise_to_structures_agg(self, batch: List[Any], num_agg: int) -> Any:
        noises = []
        for atoms in batch:
            for i in range(num_agg):
                new_atoms = deepcopy(atoms)
                positions = new_atoms.get_positions() 
                noise = np.random.normal(loc=0, scale=self.noise_std, size=positions.shape)    
                new_atoms.set_positions(positions + noise)
            noises.append(new_atoms)
        return batch, noises


    def transit(self, mass, noise_structures_batch, forces_batch, log_diffusion) -> Any:

        atoms_list = []    

        for noise_structures, forces in zip(noise_structures_batch, forces_batch):
            if self.forces_divided_by_mass:
                forces_divided_by_mass = forces / mass[:, None]
                forces_divided_by_mass_mag = forces_divided_by_mass.norm(dim=1, keepdim=True) 
                factor = torch.log(1.0 + 1000.0 * forces_divided_by_mass_mag) / forces_divided_by_mass_mag 
                value = factor * forces_divided_by_mass
            else:
                forces_mag = forces.norm(dim=1, keepdim=True)
                factor = torch.log(1.0 + 100.0 * forces_mag) / forces_mag
                value = factor * forces

            edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=noise_structures, cutoff=self.cutoff, self_interaction=True) 

            data = Data(
                pos=torch.tensor(noise_structures.get_positions(), dtype=torch.float32),
                x = value,
                lattice=torch.tensor(noise_structures.cell.array, dtype=torch.float32).unsqueeze(0), 
                edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
                edge_shift=torch.tensor(edge_shift, dtype=torch.float32),
                target = torch.tensor(log_diffusion, dtype=torch.float32)
            )

            atoms_list.append(data)

        return super().__call__(atoms_list)
    

    def __call__(self, batch: List[Any]) -> Any:
        atoms_batch = [data.x['atoms'] for data in batch]

        noise_structures_batch, _ = self.set_noise_to_structures(deepcopy(atoms_batch))

        data_list = []    

        for data in batch:

            atoms_batch = [data.x['atoms'] for _ in range(self.num_agg)]
            noise_structures_batch, _ = self.set_noise_to_structures_agg(deepcopy(atoms_batch), self.num_agg)
            forces_batch = self.properties_predictor.predict(noise_structures_batch)['forces']

            mass = data.x['atoms'].get_masses()
            log_diffusion = data.x['log_diffusion']

            graphs_batch = self.transit(mass, noise_structures_batch, forces_batch, log_diffusion)
            data_list.append(graphs_batch)
        return data_list
    
