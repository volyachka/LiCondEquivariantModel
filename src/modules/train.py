"""
This module contains functions for training and validating a model.
It includes the following functions:
- train_epoch: Trains the model for one epoch.
- validate_epoch: Validates the model for one epoch.
- train: Main training loop that trains and validates the model over multiple epochs.
"""


import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
from tqdm import tqdm
import wandb
from copy import deepcopy

def train_epoch(model, train_dataloader, optimizer, criterion, device, num_agg, predict_importance):
    """
    Train the model for one epoch.
    """
    model.train()

    total_train_loss = 0.0
    num_samples = 0

    y_true, y_pred = [], []

    for data, num_atoms in tqdm(train_dataloader):
        optimizer.zero_grad()

        data = data.to(device)
        outputs = model(data)
        prev_index = 0
        loss = 0.0
        num_samples += len(num_atoms)

        for i in range(len(num_atoms)):
            output_structures = outputs[prev_index:prev_index + num_atoms[i] * num_agg]
            prev_index += num_atoms[i] * num_agg
            if predict_importance:
                output_structures = output_structures.reshape(num_agg, num_atoms[i], 2)
                predictions, importances = output_structures[:, 0], output_structures[:, 1]
                importances = importances.reshape(num_agg, -1)
                predictions = predictions.reshape(num_agg, -1)
                importances = F.softmax(importances, dim=0)
                final_prediction = (importances * predictions).mean()
            else:
                final_prediction = outputs.mean()

            loss += criterion(final_prediction, data["target"][0])

            y_true.append(data["target"][0].unsqueeze(0))
            y_pred.append(final_prediction.unsqueeze(0)) 

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    y_true = torch.cat(y_true).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().detach().numpy()

    r2 = r2_score(y_true, y_pred)
    avg_train_loss = total_train_loss / num_samples

    return avg_train_loss, r2


def validate_epoch(model, val_dataloader, criterion, device, num_agg, predict_importance):
    """
    Validate the model for one epoch.
    """
    model.eval()

    total_val_loss = 0.0
    num_samples = 0

    y_true, y_pred = [], []

    for data, num_atoms in tqdm(val_dataloader):
        data = data.to(device)
        outputs = model(data)
        prev_index = 0
        loss = 0.0
        num_samples += len(num_atoms)

        for i in range(len(num_atoms)):
            output_structures = outputs[prev_index:prev_index + num_atoms[i] * num_agg]
            prev_index += num_atoms[i] * num_agg
            if predict_importance:
                output_structures = output_structures.reshape(num_agg, num_atoms[i], 2)
                predictions, importances = output_structures[:, 0], output_structures[:, 1]
                importances = importances.reshape(num_agg, -1)
                predictions = predictions.reshape(num_agg, -1)
                importances = F.softmax(importances, dim=0)
                final_prediction = (importances * predictions).mean()
            else:
                final_prediction = outputs.mean()

            loss += criterion(final_prediction, data["target"][0])
            y_true.append(data["target"][0].unsqueeze(0))
            y_pred.append(final_prediction.unsqueeze(0)) 

        total_val_loss += loss.item()

    y_true = torch.cat(y_true).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().detach().numpy()

    r2 = r2_score(y_true, y_pred)
    avg_val_loss = total_val_loss / num_samples

    return avg_val_loss, r2


def train(model, train_dataloader, val_dataloader, config):
    """
    Train the model for multiple epochs.
    """
    train_losses = []
    val_losses = []

    # Ensure criterion is always assigned
    criterion_name = config["training"]["criterion"]
    if criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    # Ensure optimizer is always assigned
    optimizer_name = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0)

    num_agg = config["training"]["num_agg"]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    num_epochs = config["training"]["num_epochs"]
    device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # WandB initialization
    if config["wandb"]["verbose"]:
        project_name = config["wandb"]["project_name"]
        entity = config["wandb"]["entity_name"]
        run_name = config["experiment_name"]
        wandb.init(entity=entity, project=project_name, name=run_name, config=config)

    predict_importance = config["training"]["predict_importance"]

    for epoch in range(1, num_epochs + 1):
        avg_train_loss, r2_train = train_epoch(
            model, train_dataloader, optimizer, criterion, device, num_agg, predict_importance
        )
        train_losses.append(avg_train_loss)
        avg_val_loss, r2_val = validate_epoch(model, val_dataloader, criterion, device, num_agg, predict_importance)
        val_losses.append(avg_val_loss)

        if config["wandb"]["verbose"]:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                }
            )
