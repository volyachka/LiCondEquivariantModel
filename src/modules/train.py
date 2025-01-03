"""
Trainer module for model training and validation.
Handles data loading, processing, training, and evaluation.
"""

import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import entropy
import wandb

class Trainer:
    """
    Trainer class for managing model training and validation.

    Attributes:
        model (torch.nn.Module): The neural network model.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        config (dict): Configuration dictionary.
    """

    def __init__(self, model, train_dataloader, val_dataloader, config):
        """
        Initialize the Trainer with model, data loaders, and configuration.

        Args:
            model (torch.nn.Module): The model to train.
            train_dataloader (DataLoader): Training data loader.
            val_dataloader (DataLoader): Validation data loader.
            config (dict): Training configuration.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.training_config = config["training"]
        self.device = self.training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.criterion = self._get_criterion(self.training_config["criterion"])
        self.optimizer = self._get_optimizer()

        self.num_epochs = self.training_config["num_epochs"]
        self.num_agg = self.training_config["num_agg"]
        self.predict_importance = self.training_config["predict_importance"]

        if config["wandb"]["verbose"]:
            wandb.init(
                entity=config["wandb"]["entity_name"],
                project=config["wandb"]["project_name"],
                name=config["experiment_name"],
                config=config
            )

    def _get_criterion(self, name):
        """Return the appropriate loss criterion."""
        if name == "MSELoss":
            return nn.MSELoss()
        raise ValueError(f"Unsupported criterion: {name}")

    def _get_optimizer(self):
        """Return the appropriate optimizer."""
        optimizer_name = self.training_config["optimizer"]
        learning_rate = self.training_config["learning_rate"]
        weight_decay = self.training_config.get("weight_decay", 0)

        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _process_batch(self, data, num_atoms, train=True):
        """Process a single batch of data."""
        if train:
            self.optimizer.zero_grad()

        data = data.to(self.device)
        outputs = self.model(data)
        prev_index = 0
        loss = 0.0
        y_true, y_pred = [], []
        mean_entropy = []

        for i, num_atom in enumerate(num_atoms):
            cur_slice = slice(self.num_agg * i, self.num_agg * (i + 1))
            output_structures = outputs[prev_index: prev_index + num_atom * self.num_agg]

            if self.predict_importance:
                output_structures = output_structures.reshape(self.num_agg, num_atom, 2)
                predictions, importances = output_structures[..., 0], output_structures[..., 1]
                importances = F.softmax(importances, dim=0)

                mean_importances = importances.mean(axis=0).cpu().detach().numpy()
                mean_entropy.append(entropy(mean_importances))

                final_prediction = (importances * predictions).mean()
            else:
                final_prediction = output_structures.mean()

            current_targets = data["target"][cur_slice]
            assert torch.allclose(current_targets[:1], current_targets), current_targets
            loss += self.criterion(final_prediction, current_targets[0])

            y_true.append(current_targets[0].unsqueeze(0))
            y_pred.append(final_prediction.unsqueeze(0))
            prev_index += num_atom * self.num_agg

        if train:
            loss.backward()
            self.optimizer.step()

        return loss.item(), y_true, y_pred, mean_entropy

    def _run_epoch(self, dataloader, train=True):
        """Run a single epoch of training or validation."""
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        num_samples = 0
        y_true, y_pred = [], []
        mean_entropy = []

        with torch.set_grad_enabled(train):
            for data, num_atoms in tqdm(dataloader):
                batch_loss, batch_y_true, batch_y_pred, batch_mean_entropy = (
                    self._process_batch(data, num_atoms, train=train)
                )
                total_loss += batch_loss
                num_samples += len(num_atoms)
                y_true.extend(batch_y_true)
                y_pred.extend(batch_y_pred)
                mean_entropy.extend(batch_mean_entropy)

        y_true = torch.cat(y_true).cpu().detach().numpy()
        y_pred = torch.cat(y_pred).cpu().detach().numpy()

        r2 = r2_score(y_true, y_pred)
        avg_loss = total_loss / num_samples

        return avg_loss, r2, np.array(mean_entropy).mean()

    def train_epoch(self):
        """Train for one epoch."""
        return self._run_epoch(self.train_dataloader, train=True)

    def validate_epoch(self):
        """Validate for one epoch."""
        return self._run_epoch(self.val_dataloader, train=False)

    def train(self):
        """Train the model over multiple epochs."""
        train_losses = []
        val_losses = []

        for epoch in range(1, self.num_epochs + 1):
            avg_train_loss, r2_train, entropy_train = self.train_epoch()
            train_losses.append(avg_train_loss)

            avg_val_loss, r2_val, entropy_val = self.validate_epoch()
            val_losses.append(avg_val_loss)

            if self.config["wandb"]["verbose"]:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "r2_train": r2_train,
                        "r2_val": r2_val,
                        "entropy_train": entropy_train,
                        "entropy_val": entropy_val,
                    }
                )

            if epoch % self.training_config.get("save_model_every_n_epochs", 1) == 0:
                self._save_checkpoint(epoch)

        return train_losses, val_losses

    def _save_checkpoint(self, epoch):
        """Save the model and optimizer state as a checkpoint."""
        name = self.config['experiment_name']
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f"{name}_model_epoch_{epoch}.pt")
        optimizer_path = os.path.join(output_dir, f"{name}_optimizer_epoch_{epoch}.pt")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
