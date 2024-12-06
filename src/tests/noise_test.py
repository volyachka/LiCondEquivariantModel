import numpy as np

from torch_geometric.loader import DataLoader

from src.modules.dataset import build_dataset 
from src.modules.property_prediction import SevenNetPropertiesPredictor 

from src.modules.dataset import AtomsToGraphCollater


def test_noise_sampling():
    
    dataset = build_dataset()
    checkpoint_name = '7net-0'
    SevennetPredictor = SevenNetPropertiesPredictor(checkpoint_name)

    noises = [0.01, 0.1]

    for std_true in noises:
        
        dataloader = DataLoader(dataset[:10], batch_size=1)
        dataloader.collate_fn = AtomsToGraphCollater(cutoff = 5, noise_std=std_true, properties_predictor = SevennetPredictor)
        noise_arr = []

        for i, sample in enumerate(dataloader):
            initial_coords = dataset[i].x['atoms'].get_positions().reshape(-1)
            noise_coords = sample["pos"].reshape(-1)
            noise = (noise_coords - initial_coords).detach().tolist()
            noise_arr.extend(noise)

        std_exp = np.array(noise_arr).std()

        assert np.abs((std_true - std_exp) / std_true) <= 0.1


