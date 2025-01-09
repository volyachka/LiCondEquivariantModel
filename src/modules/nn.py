"""
This module defines the SimplePeriodicNetwork class, which adapts the 
SimpleNetwork class from e3nn for use in models involving periodic boundary 
conditions. The SimplePeriodicNetwork class overrides certain methods from 
SimpleNetwork to handle periodic data and uses a mean pooling operation 
instead of summing over atom contributions.

Classes:
    SimplePeriodicNetwork: A neural network for processing periodic data 
    with modified pooling and preprocessing methods.
"""

# Standard imports
from typing import Dict, Union

# Third-party imports
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
import torch_scatter
import torch
from torch_geometric.data import Data


class SimplePeriodicNetwork(SimpleNetwork):
    """
    A network that adapts the SimpleNetwork class from e3nn to use a mean operation
    instead of summing over atom contributions per example. It also adapts the
    preprocess method for periodic boundary data.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the SimplePeriodicNetwork.

        Args:
            kwargs (dict): Keyword arguments passed to the parent SimpleNetwork class.
                - 'pool_nodes' determines whether to sum atom contributions or take the mean.
                - 'num_nodes' is set to 1.0 for each example.
        """
        # self.pool = False
        # if kwargs["pool_nodes"] is True:  # Change to `is True`
        #     kwargs["pool_nodes"] = False
        #     kwargs["num_nodes"] = 1.0
        #     self.pool = True
        # else:
        #     kwargs["num_nodes"] = 1.0
        kwargs["pool_nodes"] = False
        kwargs["num_nodes"] = 1.0
        super().__init__(**kwargs)

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Preprocesses the data for the forward pass.

        Args:
            data (Union[Data, Dict[str, torch.Tensor]]): Input data for the network.

        Returns:
            batch, x, edge_src, edge_dst, edge_vec: Processed data, including edge vectors
            for periodic boundary conditions.
        """
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        edge_src = data["edge_index"][0]  # Edge source
        edge_dst = data["edge_index"][1]  # Edge destination

        # Compute relative distances + unit cell shifts for periodic boundaries
        edge_batch = batch[edge_src]
        edge_vec = (
            data["pos"][edge_dst]
            - data["pos"][edge_src]
            + torch.einsum(
                "ni,nij->nj", data["edge_shift"], data["lattice"][edge_batch]
            )
        )

        return batch, data["x"], edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of the network. If `pool_nodes` is True, uses scatter_mean to
        aggregate the output.

        Args:
            data (Union[Data, Dict[str, torch.Tensor]]): Input data for the network.

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        output = super().forward(data)

        # if self.pool is True:  # Change to `is True`
        #     return torch_scatter.scatter_mean(
        #         output, data.batch, dim=0
        #     )  # Take mean over atoms per example
        return output
