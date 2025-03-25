"""
Trainer module for model training and validation.
Handles data loading, processing, training, and evaluation.
"""

import os
from typing import List  # Standard library imports first
from datetime import datetime
from pymatgen.core.periodic_table import Element

# Third-party imports
import numpy as np
import torch
import torch_scatter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm, trange
import wandb


def get_chemical_number(element_symbol: str) -> int:
    """
    Get the atomic number of an element given its symbol.
    """
    atomic_number = Element(element_symbol).Z
    return atomic_number


def get_element_symbol(atomic_number: int) -> str:
    """
    Get the element symbol from an atomic number.
    """
    if atomic_number == 0:
        return "conductivity_less_than_clip_value"
    element = Element.from_Z(atomic_number)
    return element.symbol.lower()


def scatter_mse_loss(target, pred, index, reduction="mean"):
    """
    Compute Scatter Mean Squared Error (MSE) Loss.

    Args:
        pred (Tensor): Predicted values (batch_size,)
        target (Tensor): Ground truth values (batch_size,)
        index (Tensor): Indices (batch_size,)
        reduction (str): Reduction type ('mean' or 'sum')

    Returns:
        Tensor: The computed loss
    """

    squared_error = (pred - target) ** 2
    scatter_squared_error = torch_scatter.scatter_sum(squared_error, index, dim=0)
    normalization = torch_scatter.scatter_sum(
        torch.ones(squared_error.shape, device=squared_error.device), index, dim=0
    )

    scatter_squared_error /= normalization

    if reduction == "mean":
        return scatter_squared_error.mean()
    if reduction == "sum":
        return scatter_squared_error.sum()
    if reduction is None:
        return scatter_squared_error
    raise NotImplementedError("Unknown type of aggregation")


class Trainer:  # pylint: disable=R0902, R0914, R0915, E0606
    """
    Trainer class for managing model training and validation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloaders: List[DataLoader],
        config: dict,
    ):
        """
        Initialize the Trainer with model, data loaders, and configuration.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders
        self.config = config

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.name = "_".join([self.config["experiment_name"], formatted_time])

        self.training_config = config["training"]
        self.prediction_strategy = self.training_config["prediction_strategy"]
        self.device = self.training_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.criterion = self._get_criterion(self.training_config["criterion"])
        self.optimizer = self._get_optimizer()

        self.num_epochs = self.training_config["num_epochs"]
        self.num_noisy_configurations = self.training_config["num_noisy_configurations"]
        self.step = 0

        modes = [
            self.training_config["softmax_within_single_atom_by_configurations"],
            self.training_config["softmax_within_single_structure_by_atoms"],
            self.training_config["softmax_within_configurations"],
        ]

        assert sum(modes) == 1
        if self.prediction_strategy in ("PerChemicalElement", "PerAtom"):
            assert (
                self.training_config["softmax_within_single_atom_by_configurations"]
                is True
            )

        if self.config["training"]["softmax_within_single_atom_by_configurations"]:
            self.mode = "softmax_within_single_atom_by_configurations"
        if self.config["training"]["softmax_within_single_structure_by_atoms"]:
            self.mode = "softmax_within_single_structure_by_atoms"
        if self.config["training"]["softmax_within_configurations"]:
            self.mode = "softmax_within_configurations"

        self.predict_importance = self.config["model"]["predict_importance"]
        # constant_lr, sequential_lr
        match self.config["scheduler"]["name"]:
            case "constant_lr":
                self.scheduler = None
            case "sequential_lr":
                warmup_lr_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=config["scheduler"]["parameterers"]["warmup_decay"],
                    total_iters=config["scheduler"]["parameterers"]["warmup_epochs"],
                )
                main_lr_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=config["training"]["num_epochs"]
                    - config["scheduler"]["parameterers"]["warmup_epochs"],
                    eta_min=0,
                )

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                    milestones=[config["scheduler"]["parameterers"]["warmup_epochs"]],
                    verbose=True,
                )
            case "plateau_lr":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.1, patience=10
                )
            case _:
                raise NotImplementedError(self.config["scheduler"]["name"])
        if config["wandb"]["verbose"]:
            tags = []
            if self.training_config["forces_divided_by_mass"]:
                tags.append("forces_divided_by_mass")
            if self.training_config["use_displacements"]:
                tags.append("use_displacements")
            if self.training_config["use_energies"]:
                tags.append("use_energies")
            if self.config["model"]["mix_properites"]:
                tags.append("mix_properites")

            tags.append(f"layers: {self.config['model']['layers']}")
            tags.append(f"radial_cutoff: {self.config['model']['radial_cutoff']}")
            tags.append(f"num_neighbors: {self.config['model']['num_neighbors']}")
            tags.append(f"number_of_basis: {self.config['model']['number_of_basis']}")
            tags.append(f"mul: {self.config['model']['mul']}")

            tags.append(self.mode)
            tags.append(self.config["data"]["upd_neigh_style"])
            tags.append(self.config["training"]["strategy_sampling"])
            tags.append(self.config["property_predictor"]["name"])
            tags.append(self.config["data"]["name"])
            tags.append(self.config["model"]["node_style_build"])

            config["model"]["num_parameters"] = sum(
                p.numel() for p in model.parameters()
            )

            self.run = wandb.init(
                entity=config["wandb"]["entity_name"],
                project=config["wandb"]["project_name"],
                name=self.name,
                tags=tags,
                config=config,
            )

    def _get_criterion(self, name):
        """Return the appropriate loss criterion."""
        if name == "MSELoss":
            return torch.nn.MSELoss()
        raise NotImplementedError(f"Unsupported criterion: {name}")

    def _get_optimizer(self):
        """Return the appropriate optimizer."""
        optimizer_name = self.config["optimizer"]["name"]
        learning_rate = self.config["optimizer"]["learning_rate"]
        weight_decay = self.config["optimizer"].get("weight_decay", 0)

        if optimizer_name == "Adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        raise NotImplementedError(f"Unsupported optimizer: {optimizer_name}")

    def _thorough_validation(self, dataloader, num_validations=5):
        y_pred_agg = []
        results = {}

        for _ in trange(num_validations):
            preds_epoch = []
            indexes_epoch = []

            with torch.set_grad_enabled(False):
                for data, num_atoms in tqdm(dataloader):

                    result = self._process_batch(data, num_atoms, train=False)

                    preds_epoch.extend(result["y_pred"])
                    indexes_epoch.extend(result["data_id"])

            preds_epoch = torch.stack(preds_epoch).cpu().detach()
            indexes_epoch = torch.stack(indexes_epoch).cpu().detach().numpy()
            indexes_epoch = np.argsort(indexes_epoch)
            y_pred_agg.append(preds_epoch[indexes_epoch])

        y_pred = torch.vstack(y_pred_agg).mean(axis=0)

        y_true = []
        mask = []
        idx_true = []

        for i, _ in dataloader:
            batch_y_true = i.y[:: self.num_noisy_configurations]
            data_id = i.idx[:: self.num_noisy_configurations]
            batch_id = i.batch[:: self.num_noisy_configurations].cpu().detach().numpy()

            if self.prediction_strategy == "PerSingleLi":
                mask.extend(np.concatenate(i.symbols[:: self.num_noisy_configurations]))
                y_true.append(batch_y_true)
                idx_true.append(data_id)

            if self.prediction_strategy == "PerChemicalElement":
                unique_symbols = np.concatenate(
                    i.symbols[:: self.num_noisy_configurations]
                )
                mask_id = 100 * batch_id + np.vectorize(get_chemical_number)(
                    unique_symbols
                )
                _, indexes = np.unique(mask_id, return_inverse=True)
                indexes = torch.tensor(indexes, device=self.device)

                batch_y_true_by_elements = torch_scatter.scatter_mean(
                    batch_y_true, indexes, dim=0
                )
                batch_y_true_id = torch_scatter.scatter_mean(
                    torch.tensor(mask_id, device=self.device), indexes, dim=0
                )
                batch_data_id = torch_scatter.scatter_mean(data_id, indexes, dim=0)
                mask.extend(batch_y_true_id.cpu().numpy())
                y_true.append(batch_y_true_by_elements)
                idx_true.append(batch_data_id)

            if self.prediction_strategy == "PerSingleLi":
                batch_data_id = torch_scatter.scatter_mean(batch_id, indexes, dim=0)
                y_true.append(batch_y_true)
                idx_true.append(batch_data_id)

        y_true = torch.concatenate(y_true).cpu().detach()

        idx_true = torch.concatenate(idx_true).cpu().detach()

        y_true = y_true[np.argsort(idx_true)]

        results["thorough_r2"] = r2_score(y_true, y_pred)
        results["thorough_loss"] = mean_squared_error(y_true, y_pred)
        results["y_true"] = y_true
        results["y_pred"] = y_pred

        if self.prediction_strategy != "PerSingleLi":
            mask = np.stack(mask)
            mask = mask[np.argsort(idx_true)]
            if self.prediction_strategy == "PerAtom":
                li_mask = mask == "Li"
            elif self.prediction_strategy == "PerChemicalElement":
                li_mask = mask % 100 == 3

            li_true = y_true[li_mask]
            li_pred = y_pred[li_mask]
            results["li_true"] = li_true.numpy()
            results["li_pred"] = li_pred.numpy()
            results["li_thorough_loss"] = self.criterion(li_pred, li_true)
            results["li_thorough_r2"] = r2_score(li_true.numpy(), li_pred.numpy())

        return results

    def _generate_index_arrays(self, num_atoms):

        indexes = torch.arange(len(num_atoms) * self.num_noisy_configurations)
        repeats = torch.tensor(num_atoms).repeat_interleave(
            self.num_noisy_configurations
        )
        indexing_noise_variations = indexes.repeat_interleave(repeats).to(self.device)

        indexing_atoms = []
        last_idx = 0
        for num in num_atoms:
            indexing_atoms.append(
                torch.arange(last_idx, last_idx + num).repeat(
                    self.num_noisy_configurations
                )
            )
            last_idx += num
        indexing_atoms = torch.cat(indexing_atoms).to(self.device)

        indexes = torch.arange(len(num_atoms))
        repeats = torch.tensor(num_atoms) * self.num_noisy_configurations
        indexing_both = indexes.repeat_interleave(repeats).to(self.device)

        return indexing_atoms, indexing_noise_variations, indexing_both

    def _choose_index(self, indexing_atoms, indexing_noise_variations, indexing_both):
        if self.mode == "softmax_within_single_atom_by_configurations":
            return indexing_atoms
        if self.mode == "softmax_within_single_structure_by_atoms":
            return indexing_noise_variations
        if self.mode == "softmax_within_configurations":
            return indexing_both
        raise NotImplementedError("Unknown type of aggregation")

    def _calculate_entropy(self, importances, aggregation_index):
        entropy = torch_scatter.scatter_sum(
            -importances * torch.log(importances), aggregation_index, dim=0
        )
        normalization = torch.log(
            torch_scatter.scatter_sum(
                torch.ones(importances.shape, device=self.device),
                aggregation_index,
                dim=0,
            )
        )
        entropy /= normalization
        return entropy

    def _process_batch(
        self, data, num_atoms, train=True
    ):  # pylint: disable=R0912, R0914
        """Process a single batch of data."""
        if train:
            self.optimizer.zero_grad()

        indexing_atoms, indexing_noise_variations, indexing_both = (
            self._generate_index_arrays(num_atoms)
        )
        aggregation_index = self._choose_index(
            indexing_atoms, indexing_noise_variations, indexing_both
        )

        data = data.to(self.device)
        assert torch.isnan(data["x"]).any().item() is False

        if self.config["model"]["mix_properites"]:
            outputs = self.model(data, indexing_noise_variations)
        else:
            outputs = self.model(data)
        if self.predict_importance:
            predictions, importances = (
                outputs[..., 0],
                outputs[..., 1],
            )
        else:
            predictions = outputs

        if self.predict_importance:
            importances = torch_scatter.scatter_softmax(
                importances, aggregation_index, dim=0
            )

            assert all(num > 1 for num in num_atoms)

            entropy = self._calculate_entropy(importances, aggregation_index)
        else:
            entropy = -torch.ones(len(num_atoms))

        if self.predict_importance:
            intermediate_predictions = torch_scatter.scatter_sum(
                importances * predictions, aggregation_index, dim=0
            )
        else:
            intermediate_predictions = torch_scatter.scatter_mean(
                predictions, aggregation_index, dim=0
            )
        idx_1, _ = torch_scatter.scatter_min(indexing_both, aggregation_index, dim=0)
        idx_2, _ = torch_scatter.scatter_max(indexing_both, aggregation_index, dim=0)

        assert torch.equal(idx_1, idx_2)

        if self.prediction_strategy in ("PerChemicalElement", "PerAtom"):
            final_predictions = intermediate_predictions.squeeze()
        else:
            final_predictions = torch_scatter.scatter_mean(
                intermediate_predictions, idx_1, dim=0
            )
            if not self.predict_importance:
                final_predictions = final_predictions.squeeze()

        y_true = data["y"][:: self.num_noisy_configurations]
        batch_id = (
            data["batch"][:: self.num_noisy_configurations].cpu().detach().numpy()
        )
        y_pred = final_predictions.detach()

        assert final_predictions.shape == y_true.shape

        unique_symbols = np.concatenate(
            data["symbols"][:: self.num_noisy_configurations]
        )

        result = {}
        if self.prediction_strategy == "PerChemicalElement":
            mask = 100 * batch_id + np.vectorize(get_chemical_number)(unique_symbols)
            # это нужно потому что в mask принадлежит очень большой интервал и куча нулевых позиций
            elements_values, indexes = np.unique(mask, return_inverse=True)
            indexes = torch.tensor(indexes, device=self.device)
            y_pred_by_elements = torch_scatter.scatter_mean(
                final_predictions, indexes, dim=0
            )
            y_true_by_elements = torch_scatter.scatter_mean(y_true, indexes, dim=0)
            y_true_id = torch_scatter.scatter_mean(
                torch.tensor(mask, device=self.device), indexes, dim=0
            )

            result["y_true"] = y_true_by_elements.detach()
            result["y_pred"] = y_pred_by_elements.detach()
            result["loss"] = self.criterion(y_true_by_elements, y_pred_by_elements)
            result["data_id"] = y_true_id
            result["elements_values"] = elements_values

        elif self.prediction_strategy == "PerAtom":
            result["loss"] = self.criterion(y_true, final_predictions)
            result["y_true"] = y_true
            result["y_pred"] = y_pred
            result["mask"] = unique_symbols

        result["entropy"] = entropy
        # result["unique_symbols"] = unique_symbols
        # result["elements_values"] = elements_values
        # result["indexes"] = indexes
        return result

    def _run_epoch(self, dataloader, train=True):  # pylint: disable=R0912, R0914
        """Run a single epoch of training or validation."""
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        num_samples = 0
        y_true, y_pred = [], []

        if self.prediction_strategy in ("PerChemicalElement", "PerAtom"):
            mask = []

        entropy = []

        with torch.set_grad_enabled(train):
            for data, num_atoms in tqdm(dataloader):
                result = self._process_batch(data, num_atoms, train=train)
                batch_loss = result["loss"]
                # unique_symbols = result["unique_symbols"]
                # indexes = result["elements_values"]
                batch_entropy = result["entropy"]

                batch_y_true = result["y_true"]
                batch_y_pred = result["y_pred"]

                if self.prediction_strategy == "PerChemicalElement":
                    total_loss += batch_loss.item() * len(batch_y_true)
                    num_samples += len(batch_y_true)
                    mask.extend(result["elements_values"])

                elif self.prediction_strategy == "PerAtom":
                    batch_y_true = result["y_true"]
                    batch_y_pred = result["y_pred"]
                    total_loss += batch_loss.item() * len(batch_y_true)
                    num_samples += len(batch_y_true)

                    mask.extend(result["mask"])

                    # li_mask = unique_symbols == "Li"
                    # li_true.extend(batch_y_true[li_mask])
                    # li_pred.extend(batch_y_pred[li_mask])

                elif self.prediction_strategy == "PerSingleLi":
                    batch_y_true = result["y_true"]
                    batch_y_pred = result["y_pred"]
                    total_loss += batch_loss.item() * len(num_atoms)
                    num_samples += len(num_atoms)

                if train:
                    if self.config["wandb"]["verbose"]:
                        batch_r2 = r2_score(
                            batch_y_true.cpu().numpy(), batch_y_pred.cpu().numpy()
                        )

                        info = {
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "batch_loss": batch_loss.item(),
                            "batch_r2": batch_r2,
                        }

                        wandb.log(info, step=self.step)

                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    if self.step % 100 == 0 and self.config["wandb"]["verbose"]:
                        val_results = {}
                        info = {}
                        for name, val_dataloader in tqdm(self.val_dataloaders.items()):
                            val_results[name] = self.validate_epoch(val_dataloader)

                        for name, result in tqdm(val_results.items()):
                            info = self._update_logging_info(name, result, info)
                            wandb.log(info, step=self.step)

                y_true.extend(batch_y_true)
                y_pred.extend(batch_y_pred)
                entropy.extend(batch_entropy)

                if train:
                    self.step += 1

        y_true = torch.stack(y_true).cpu().detach()
        y_pred = torch.stack(y_pred).cpu().detach()
        entropy = torch.stack(entropy).cpu().detach().numpy()

        if self.prediction_strategy != "MSELossPerSingleLi":
            mask = np.stack(mask)

        results = {}

        results["loss"] = total_loss / num_samples
        results["r2"] = r2_score(y_true.numpy(), y_pred.numpy())
        results["mean_entropy"] = entropy.mean()

        if self.prediction_strategy != "PerSingleLi":
            if self.prediction_strategy == "PerAtom":
                li_mask = mask == "Li"
            elif self.prediction_strategy == "PerChemicalElement":
                li_mask = mask % 100 == 3

            li_true = y_true[li_mask]
            li_pred = y_pred[li_mask]
            results["li_true"] = li_true.numpy()
            results["li_pred"] = li_pred.numpy()
            results["li_loss"] = self.criterion(li_pred, li_true)
            results["li_r2"] = r2_score(li_true.numpy(), li_pred.numpy())

        return results

    def train_epoch(self):
        """Train for one epoch."""
        return self._run_epoch(self.train_dataloader, train=True)

    def validate_epoch(self, val_dataloader=None):
        """Validate for one epoch."""
        return self._run_epoch(val_dataloader, train=False)

    def _log_scatter_plot(self, name, y_true, y_pred, title):
        table = wandb.Table(
            data=[[label, prediction] for label, prediction in zip(y_true, y_pred)],
            columns=["label", "prediction"],
        )
        wandb.log(
            {
                f"plot_{name}": wandb.plot.scatter(
                    table, "label", "prediction", title=title
                )
            },
            step=self.step,
        )

    def _update_thorough_results(self, name, results, info):

        # if self.predict_per_atom:

        #     for element_symbol in ["conductivity_less_than_clip_value", "li"]:
        #         info.update(
        #             {
        #                 f"{element_symbol}_thorough_{name}_loss":
        #                   results[f"{element_symbol}_thorough_loss"],
        #                 f"{element_symbol}_thorough_{name}_r2":
        #                   results[f"{element_symbol}_thorough_r2"],
        #             }
        #         )

        #         self._log_scatter_plot(
        #             f"{element_symbol}_{name}",
        #             results[f"{element_symbol}_true"],
        #             results[f"{element_symbol}_pred"],
        #             f"{element_symbol}_{name} (per atom)",
        #         )

        # else:

        info.update(
            {
                f"thorough_{name}_loss": results["thorough_loss"],
                f"thorough_{name}_r2": results["thorough_r2"],
            }
        )

        self._log_scatter_plot(name, results["y_true"], results["y_pred"], name)

        if self.prediction_strategy in ("PerChemicalElement", "PerAtom"):
            info.update(
                {
                    f"li_thorough_{name}_loss": results["li_thorough_loss"],
                    f"li_thorough_{name}_r2": results["li_thorough_r2"],
                }
            )

    def _construct_logging_info_from_thorough_validation(self, info):
        thorough_train_results = self._thorough_validation(self.train_dataloader)
        self._update_thorough_results("train", thorough_train_results, info)

        for name, dataloader in self.val_dataloaders.items():
            thorough_val_results = self._thorough_validation(dataloader)
            self._update_thorough_results(name, thorough_val_results, info)

        return info

    def _update_logging_info(self, prefix, results, info):
        info.update(
            {
                f"{prefix}_loss": results["loss"],
                f"{prefix}_r2": results["r2"],
            }
        )

        if self.prediction_strategy in ("PerChemicalElement", "PerAtom"):
            info.update(
                {
                    f"li_{prefix}_loss": results["li_loss"],
                    f"li_{prefix}_r2": results["li_r2"],
                }
            )

        # if self.predict_importance:
        #     info[f"{prefix}_entropy"] = results["mean_entropy"]

        # if self.predict_per_atom:
        #     info.update(
        #         {
        #             f"li_{prefix}_loss": results["li_loss"],
        #             f"li_{prefix}_r2": results["li_r2"],
        #         }
        #     )
        return info

    def _construct_logging_info(self, epoch, train_results, val_results):
        info = {"epoch": epoch}
        info = self._update_logging_info("train", train_results, info)

        for name, result in val_results.items():
            info = self._update_logging_info(name, result, info)

        return info

    def train(self):
        """Train the model over multiple epochs."""
        train_losses, val_losses = [], []
        val_results = {}
        for epoch in range(1, self.num_epochs + 1):
            train_results = self.train_epoch()
            for name, dataloader in self.val_dataloaders.items():
                val_results[name] = self.validate_epoch(dataloader)
            if self.config["wandb"]["verbose"]:
                info = self._construct_logging_info(epoch, train_results, val_results)
                if (
                    epoch == 1
                    or epoch % self.training_config.get("save_model_every_n_epochs", 1)
                    == 0
                ):
                    info = self._construct_logging_info_from_thorough_validation(info)

                wandb.log(info, step=self.step)

            if epoch % self.training_config.get("save_model_every_n_epochs", 1) == 0:
                self._save_checkpoint(epoch)

        return train_losses, val_losses

    def _save_checkpoint(self, epoch):
        """Save the model and optimizer state as a checkpoint."""
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f"{self.name}_model_epoch_{epoch}.pt")
        optimizer_path = os.path.join(
            output_dir, f"{self.name}_optimizer_epoch_{epoch}.pt"
        )

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
