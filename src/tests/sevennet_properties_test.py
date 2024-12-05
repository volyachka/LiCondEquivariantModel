import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from src.modules.dataset import build_dataset
from src.modules.property_prediction import SevenNetPropertiesPredictor

from sevenn.sevennet_calculator import SevenNetCalculator

def test_sevennet_properties():
    
    dataset = build_dataset()

    checkpoint_name = '7net-0'

    SevenNetCalc = SevenNetCalculator(checkpoint_name, device='cpu')
    SevennetPredictor = SevenNetPropertiesPredictor(checkpoint_name)


    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch in dataloader:   
        structures = batch.x['atoms']
        for atoms in structures:
            positions = atoms.get_positions() 
            noise = np.random.normal(loc=0, scale=0.01, size=positions.shape)    
            atoms.set_positions(positions + noise)
        
        properties_batch = SevennetPredictor.predict(structures)

        for atoms, forces, energy in zip(structures, properties_batch['forces'], properties_batch['energy']):
            atoms.calc = SevenNetCalc
            forces_from_calc = atoms.get_forces()
            energy_from_calc = atoms.get_potential_energy()

            assert np.isclose(forces_from_calc, forces.detach().numpy(), atol=1e-6, rtol=1e-5).all()
            assert np.isclose(energy_from_calc, energy.detach().numpy(), atol=1e-6, rtol=1e-5).all()

