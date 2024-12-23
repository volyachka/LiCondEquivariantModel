import torch

from modules.dataset import AtomsToGraphCollater, build_dataloader_cv
from modules.nn import SimplePeriodicNetwork
from modules.property_prediction import SevenNetPropertiesPredictor
from modules.train import train

import os
import argparse
import yaml

def load_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help="Path to the configuration YAML file.")
    
    args = parser.parse_args()
    config = load_config(args.config)

    if config['training']['device'] == "" or config['training']['device'] is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['training']['device']

    if config['property_predictor']['name'] == 'sevennet':
        checkpoint_name = config['property_predictor']['checkpoint']
        SevennetPredictor = SevenNetPropertiesPredictor(checkpoint_name, device)

    train_dataloader, val_dataloader = build_dataloader_cv(config)
    
    train_dataloader.collate_fn = AtomsToGraphCollater(cutoff = config['training']['radial_cutoff'], 
                                                       noise_std = config['data']['noise_std'], 
                                                       properties_predictor = SevennetPredictor, 
                                                       forces_divided_by_mass = config['training']['forces_divided_by_mass'],
                                                       num_agg = config['training']['num_agg'])
    
    val_dataloader.collate_fn = AtomsToGraphCollater(cutoff = config['training']['radial_cutoff'], 
                                                     noise_std=config['data']['noise_std'], 
                                                     properties_predictor = SevennetPredictor, 
                                                     forces_divided_by_mass = config['training']['forces_divided_by_mass'],
                                                     num_agg = config['training']['num_agg'])

    net = SimplePeriodicNetwork(
        irreps_in="1x1o",  
        irreps_out="2x0e",  # Single scalar (L=0 and even parity) to output (for example) energy
        max_radius=config['training']['radial_cutoff'], # Cutoff radius for convolution
        num_neighbors=config['training']['num_neighbors'],  # scaling factor based on the typical number of neighbors
        pool_nodes=True,  # We pool nodes to predict total energy
    )


    model = train(net, train_dataloader, val_dataloader, config)

    name = config['experiment_name']
    PATH = os.path.join(config['output_dir'], f'{name}.pt')
    torch.save(model.state_dict(), PATH)