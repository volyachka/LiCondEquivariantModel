import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
from tqdm import tqdm
import wandb

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, train_dataloader, optimizer, criterion, device):
    model.train()  
    total_train_loss = 0
    for data in tqdm(train_dataloader):
        optimizer.zero_grad()

        data = data.to(device)
        outputs = model(data).squeeze()
        loss = criterion(outputs, data['target'])
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss

def validate_epoch(model, val_dataloader, criterion, device):
    model.eval()  
    total_val_loss = 0

    for data in val_dataloader:
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data).squeeze().detach()
        loss = criterion(outputs, data['target'])
        total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss

def train(model, train_dataloader, val_dataloader, config):
    train_losses = []
    val_losses = []


    criterion_name = config['training']['criterion']
    if criterion_name == "MSELoss":
        criterion = nn.MSELoss()

    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = config['training']['num_epochs']

    if config['training']['device'] == "" or config['training']['device'] is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    project_name = config['wandb']['project_name']
    run_name = config['experiment_name']
    if project_name is not None:

        wandb.init(project=project_name, name = run_name, config=config)

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate_epoch(model, val_dataloader, criterion, device)
        val_losses.append(avg_val_loss)

        if project_name is not None:
            wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss
                    })
            
        if config['training']['verbose'] == True:

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

            plt.figure(figsize=(15, 10))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, epoch + 2), train_losses, label='Training Loss', color='blue')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='red')
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        
        clear_output(True)
        
    if project_name is not None:
        wandb.finish()

    return model