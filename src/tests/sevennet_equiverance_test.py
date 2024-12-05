import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from src.modules.dataset import build_dataset 
from src.modules.property_prediction import SevenNetPropertiesPredictor 

from copy import deepcopy
import torch
import e3nn.o3 as o3


def test_sevennet_equivariance():

    dataset = build_dataset()
    checkpoint_name = '7net-0'
    SevennetPredictor = SevenNetPropertiesPredictor(checkpoint_name)

    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    
    for batch in dataloader:
        structures = batch.x['atoms']
        initial_atoms_arr = []
        rotated_atoms_arr = []
        wigner_matrices_arr = []

        for atoms in structures:

            positions = atoms.get_positions() 
            noise = np.random.normal(loc=0, scale=0.01, size=positions.shape)    

            #add noise
            atoms.set_positions(positions + noise)
            atoms_rotated = deepcopy(atoms)
        
            positions = atoms_rotated.get_positions() 
            angles = torch.rand(3) * np.pi * 2
            wigner_D = o3.wigner_D(1, *angles).detach().numpy()

            fractional_shift = np.random.rand(1, 3)
            cartesian_shift = np.dot(fractional_shift, atoms.cell)

            #add shift and rotate
            rotated_positions = (positions + cartesian_shift) @ wigner_D.T

            atoms_rotated.cell = atoms_rotated.cell @ wigner_D.T
            atoms_rotated.set_positions(rotated_positions)

            initial_atoms_arr.append(atoms)
            rotated_atoms_arr.append(atoms_rotated)
            wigner_matrices_arr.append(wigner_D)
            
        #energy check
        assert np.allclose(SevennetPredictor.predict(initial_atoms_arr)['energy'], SevennetPredictor.predict(rotated_atoms_arr)['energy'])

        #forces check
        for force_initial, force_rotated, wigner_D in zip(SevennetPredictor.predict(initial_atoms_arr)['forces'], SevennetPredictor.predict(rotated_atoms_arr)['forces'], wigner_matrices_arr):
            assert np.allclose(force_initial @ wigner_D.T, force_rotated, atol=1e-3, rtol=1e-6)

        print('all assertions passed')