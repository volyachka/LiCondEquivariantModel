"""
Trainer module for model training and validation.
Handles data loading, processing, training, and evaluation.
"""

import os
import numpy as np
import torch
import torch_scatter
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
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
        self.device = self.training_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.pool_nodes = config["predict_importance"]["pool_nodes"]
        self.criterion = self._get_criterion(self.training_config["criterion"])
        self.optimizer = self._get_optimizer()

        self.num_epochs = self.training_config["num_epochs"]
        self.num_noisy_configurations = self.training_config["num_noisy_configurations"]

        self.softmax_within_single_structure_by_configurations = self.config["predict_importance"][
            "softmax_within_single_structure_by_configurations"
        ]
        self.softmax_within_single_atom_by_configurations = self.config["predict_importance"][
            "softmax_within_single_atom_by_configurations"
        ]
        self.softmax_within_configurations = self.config["predict_importance"][
            "softmax_within_configurations"
        ]

        if config["wandb"]["verbose"]:
            tags = []
            if self.training_config["forces_divided_by_mass"]:
                tags.append("forces_divided_by_mass")
            if self.training_config["use_displacements"]:
                tags.append("use_displacements")
            if self.training_config["use_energies"]:
                tags.append("use_energies")
            if self.config["predict_importance"]["pool_nodes"]:
                tags.append("pool_nodes")
            if self.config["predict_importance"]["softmax_within_single_structure_by_configurations"]:
                tags.append("softmax_within_single_structure_by_configurations")
            if self.config["predict_importance"]["softmax_within_single_atom_by_configurations"]:
                tags.append("softmax_within_single_atom_by_configurations")
            if self.config["predict_importance"]["softmax_within_configurations"]:
                tags.append("softmax_within_configurations")
                                    
            wandb.init(
                entity=config["wandb"]["entity_name"],
                project=config["wandb"]["project_name"],
                name=config["experiment_name"],
                tags=tags,
                config=config,
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
            return optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _thorough_validation(self, dataloader):
        y_pred_agg = []
        for _ in range(20):
            preds = []

            with torch.set_grad_enabled(False):
                for data, num_atoms in tqdm(dataloader):
                    batch_loss, batch_y_true, batch_y_pred, batch_entropy = (
                        self._process_batch(data, num_atoms, train=train)
                    )
                    total_loss += batch_loss
                    num_samples += len(num_atoms)
                    preds.extend(batch_y_pred)
            y_pred_agg.append(preds)

        y_pred = (
            torch.stack([torch.concatenate(p) for p in y_pred_agg], dim=0)
            .mean(axis=0)
            .cpu()
            .detach()
            .numpy()
        )
        
        y_true = (
            torch.concatenate([i.target.cpu() for i, _ in dataloader])
            .cpu()
            .detach()
            .numpy()
        )

        val_thorough_r2 = r2_score(y_true, y_pred)
        val_thorough_loss = mean_squared_error(y_true, y_pred)

        return val_thorough_loss, val_thorough_r2

    def _process_batch(self, data, num_atoms, train=True):
        """Process a single batch of data."""
        if train:
            self.optimizer.zero_grad()

        data = data.to(self.device)
        outputs = self.model(data)
        predictions, importances = (
            outputs[..., 0],
            outputs[..., 1],
        )
        loss = 0.0
        y_true, y_pred = [], []

        if self.softmax_within_single_structure_by_configurations:
            indexes = torch.arange(len(num_atoms) * self.num_noisy_configurations)
            repeats = torch.tensor(num_atoms).repeat_interleave(self.num_noisy_configurations)
            idx = indexes.repeat_interleave(repeats).to(self.device)

            importances = torch_scatter.scatter_softmax(importances, idx, dim=0)
            entropy = torch_scatter.scatter_sum(
                -importances * torch.log(importances), idx, dim=0
            )
            normalization = (
                torch.log(torch.tensor(num_atoms))
                .repeat_interleave(self.num_noisy_configurations)
                .to(self.device)
            )
            entropy /= normalization
            assert entropy.shape[0] == self.num_noisy_configurations * len(data)
            intermediate_predictions = torch_scatter.scatter_mean(
                importances * predictions, idx, dim=0
            )
            idx_agg = (
                torch.arange(len(num_atoms))
                .repeat_interleave(self.num_noisy_configurations)
                .to(self.device)
            )
            final_predictions = torch_scatter.scatter_mean(
                intermediate_predictions, idx_agg, dim=0
            )

        if self.softmax_within_single_atom_by_configurations:
            idx = []
            last_idx = 0
            for num in num_atoms:
                idx.append(
                    torch.arange(last_idx, last_idx + num).repeat(self.num_noisy_configurations)
                )
                last_idx += num
            idx = torch.cat(idx).to(self.device)

            importances = torch_scatter.scatter_softmax(importances, idx, dim=0)
            entropy = torch_scatter.scatter_sum(
                -importances * torch.log(importances), idx, dim=0
            )
            normalization = torch.log(torch.tensor(self.num_noisy_configurations, dtype = torch.float32))
            entropy /= normalization
            assert entropy.shape[0] == sum(num_atoms)
            intermediate_predictions = torch_scatter.scatter_mean(
                importances * predictions, idx, dim=0
            )

            idx_atoms = (
                torch.arange(len(num_atoms))
                .repeat_interleave(torch.tensor(num_atoms))
                .to(self.device)
            )
            final_predictions = torch_scatter.scatter_mean(
                intermediate_predictions, idx_atoms, dim=0
            )

        if self.softmax_within_configurations:
            indexes = torch.arange(len(num_atoms))
            repeats = torch.tensor(num_atoms) * self.num_noisy_configurations
            idx = indexes.repeat_interleave(repeats).to(self.device)

            importances = torch_scatter.scatter_softmax(importances, idx, dim=0)

            entropy = torch_scatter.scatter_sum(
                -importances * torch.log(importances), idx, dim=0
            )
            normalization = torch.log(torch.tensor(num_atoms) * self.num_noisy_configurations).to(
                self.device
            )
            entropy /= normalization
            assert entropy.shape[0] == len(num_atoms)
            final_predictions = torch_scatter.scatter_mean(
                importances * predictions, idx, dim=0
            )

        if self.pool_nodes:
            idx = (
                torch.arange(len(num_atoms))
                .repeat_interleave(self.num_noisy_configurations)
                .to(self.device)
            )
            importances = torch_scatter.scatter_softmax(importances, idx, dim=0)
            entropy = torch_scatter.scatter_sum(
                -importances * torch.log(importances), idx, dim=0
            )

            normalization = torch.log(torch.tensor(self.num_noisy_configurations, dtype = torch.float32))

            final_predictions = torch_scatter.scatter_mean(
                importances * predictions, idx, dim=0
            )

            assert entropy.shape[0] == len(data)
            
        y_true = data["target"][:: self.num_noisy_configurations]
        y_pred = final_predictions.detach()

        loss += self.criterion(final_predictions, y_true)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss.item(), y_true, y_pred, entropy

    def _run_epoch(self, dataloader, train=True):
        """Run a single epoch of training or validation."""
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        num_samples = 0
        y_true, y_pred = [], []
        entropy = []

        with torch.set_grad_enabled(train):
            for data, num_atoms in tqdm(dataloader):
                batch_loss, batch_y_true, batch_y_pred, batch_entropy = (
                    self._process_batch(data, num_atoms, train=train)
                )
                total_loss += batch_loss
                num_samples += len(num_atoms)
                y_true.extend(batch_y_true)
                y_pred.extend(batch_y_pred)
                entropy.extend(batch_entropy)

        y_true = torch.stack(y_true).cpu().detach().numpy()
        y_pred = torch.stack(y_pred).cpu().detach().numpy()
        entropy = torch.stack(entropy).cpu().detach().numpy()

        r2 = r2_score(y_true, y_pred)
        avg_loss = total_loss / num_samples

        return avg_loss, r2, entropy.mean()

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
                        "entropy_train": entropy_val,
                    }
                )

                if (
                    self.config["training"]["num_noisy_configurations"] == 1
                    and epoch % self.training_config.get("save_model_every_n_epochs", 1)
                    == 0
                ):
                    thorough_train_loss, thorough_train_r2 = self._thorough_validation(
                        self.train_dataloader
                    )
                    thorough_val_loss, thorough_val_r2 = self._thorough_validation(
                        self.val_dataloader
                    )
                    wandb.log(
                        {
                            "thorough_val_loss": thorough_val_loss,
                            "thorough_val_r2": thorough_val_r2,
                            "thorough_train_loss": thorough_train_loss,
                            "thorough_train_r2": thorough_train_r2,
                        }
                    )

        return train_losses, val_losses

    def _save_checkpoint(self, epoch):
        """Save the model and optimizer state as a checkpoint."""
        name = self.config["experiment_name"]
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f"{name}_model_epoch_{epoch}.pt")
        optimizer_path = os.path.join(output_dir, f"{name}_optimizer_epoch_{epoch}.pt")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
