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
import torch
from torch_geometric.data import Data
from torch_scatter import scatter

# e3nn imports
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2103.gate_points_message_passing import MessagePassing


# pylint: disable=R0902, R0913, R0917
class SimplePeriodicNetwork(torch.nn.Module):
    """
    A neural network for processing periodic data using the e3nn framework.

    This class is a modified version of SimpleNetwork, specifically designed to
    handle periodic boundary conditions. It overrides certain methods from
    SimpleNetwork, including pooling strategies, to accommodate the periodic
    nature of the input data.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        max_radius,
        num_neighbors: int,
        num_nodes: int,
        number_of_basis: int,
        mul: int,
        layers: int,
        lmax: int,
        fc_neurons: list,
        pool_nodes: bool,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        assert pool_nodes is False
        assert self.num_nodes == 1
        assert fc_neurons[0] == self.number_of_basis
        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )

        self.mp = MessagePassing(
            irreps_node_input=irreps_in,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_node_attr="0e",
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            layers=layers,
            fc_neurons=fc_neurons,
            num_neighbors=num_neighbors,
        )

        self.irreps_in = self.mp.irreps_node_input
        self.irreps_out = self.mp.irreps_node_output

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Preprocesses the data for the forward pass.

        Args:
            data (Union[Data, Dict[str, torch.Tensor]]): Input data for the network.

        Returns:
            batch, x, edge_src, edge_dst, edge_vec: Processed data, including edge vectors
            for periodic boundary conditions.
        """

        batch = data["batch"]

        edge_src = data["edge_index"][0]  # Edge source
        edge_dst = data["edge_index"][1]  # Edge destination

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
        Perform a forward pass through the network.

        Args:
            data (Union[Data, Dict[str, torch.Tensor]]): The input data containing
            the node features,
            edge attributes, and other necessary information for the model.

        Returns:
            torch.Tensor: The output of the network, which could either be a node feature tensor or
            a pooled graph feature tensor, depending on whether pooling is enabled.
        """
        batch, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="cosine",  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = node_inputs.new_ones(node_inputs.shape[0], 1)

        node_outputs = self.mp(
            node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
        )

        if self.pool_nodes:
            return scatter(node_outputs, batch, dim=0).div(self.num_nodes**0.5)
        return node_outputs
