import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch

import sevenn
from dataset import AtomsToGraphCollater, build_dataset
from nn import SimplePeriodicNetwork
from property_prediction import SevenNetPropertiesPreditcor
from train import train


if __name__ == "__main__":

    df = pd.read_csv('sevennet_slopes.csv')
    df['v1_Li_slope'] = df['v1_Li_slope'].clip(lower=1e-4)
    dataset = build_dataset(df, temp = 1000)

    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    batch_size = 10

    checkpoint = sevenn.util.pretrained_name_to_path('7net-0')
    sevennet_model, sevennet_config = sevenn.util.model_from_checkpoint(checkpoint)

    checkpoint_name = '7net-0'
    sevennet_predictor = SevenNetPropertiesPreditcor(checkpoint_name)
 

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    train_dataloader.collate_fn = AtomsToGraphCollater(cutoff = 5, noise_std=0.01, properties_predictor = sevennet_predictor)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    val_dataloader.collate_fn = AtomsToGraphCollater(cutoff = 5, noise_std=0.01, properties_predictor = sevennet_predictor)

    radial_cutoff = 5

    net = SimplePeriodicNetwork(
        irreps_in="1x1o",  
        irreps_out="1x0e",  # Single scalar (L=0 and even parity) to output (for example) energy
        max_radius=radial_cutoff, # Cutoff radius for convolution
        num_neighbors=10.0,  # scaling factor based on the typical number of neighbors
        pool_nodes=True,  # We pool nodes to predict total energy
    )

    criterion = nn.MSELoss()  # Example: Mean Squared Error
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 50
    # project_name = 'LiCondEquivariantModel'
    project_name = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(net, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, device, verbose = False, project_name = project_name)