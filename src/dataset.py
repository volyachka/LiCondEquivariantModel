from typing import Any, List, Optional, Sequence
import ase.io
from sevenn.sevennet_calculator import SevenNetCalculator

from copy import deepcopy
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data
import numpy as np
from utils import query_mpid_structure
from tqdm import tqdm
import torch
from torch_geometric.loader.dataloader import Collater

def build_dataset(df, temp):

    mpids = df[df['temperature'] == temp]['mpid'].to_list()
    docs = query_mpid_structure(mpids=mpids)

    dataset = []
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
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        super().__init__([], follow_batch, exclude_keys)
        self.cutoff = cutoff
        self.noise_std = noise_std
        self.properties_predictor = properties_predictor

    def set_noise_to_structures(self, batch: List[Any]) -> Any:
        for data in batch:
            atoms = data.x['atoms']
            positions = atoms.get_positions() 
            noise = np.random.normal(loc=0, scale=self.noise_std, size=positions.shape)    
            atoms.set_positions(positions + noise)
            
        return batch

    def __call__(self, batch: List[Any]) -> Any:

        noise_structures_batch = self.set_noise_to_structures(deepcopy(batch))
        properties = self.properties_predictor.predict(noise_structures_batch)

        atoms_list = []    

        for data, forces in zip(noise_structures_batch, properties['forces']):
            forces_mag = forces.norm(dim=1, keepdim=True)
            factor = torch.log(1.0 + 100.0 * forces_mag) / forces_mag
            atoms = data.x['atoms']
            log_diffusion = data.x['log_diffusion']
            edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=atoms, cutoff=self.cutoff, self_interaction=True) 

            data = Data(
                pos=torch.tensor(atoms.get_positions(), dtype=torch.float32),
                x=forces * factor,
                lattice=torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0),  # We add a dimension for batching
                edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
                edge_shift=torch.tensor(edge_shift, dtype=torch.float32),
                target = torch.tensor(log_diffusion, dtype=torch.float32)
            )

            atoms_list.append(data)

        return super().__call__(atoms_list)
