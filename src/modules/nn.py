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
import torch_scatter
from e3nn import o3


from e3nn.nn.models.v2103.points_convolution import Convolution
from e3nn.nn.models.v2103.gate_points_message_passing import SingleMessagePassing
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
from e3nn.math import soft_one_hot_linspace


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

        assert kwargs["pool_nodes"] is False
        assert kwargs["num_nodes"] == 1.0
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

        return (
            data["batch"],
            data["x"],
            data["edge_src"],
            data["edge_dst"],
            data["edge_vec"],
        )

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


# pylint: disable=R0913, R0902, R0917
class MixingNetwork(torch.nn.Module): 
    """
    A network that adapts the SimpleNetwork functional with possibility to share 
    features between message passing iterations
    """

    def __init__(
        self,
        layers,
        irreps_in,
        irreps_out,
        pool_nodes,
        max_radius,
        num_neighbors,
        num_nodes,
        mul: int = 50,
        lmax: int = 2,
        number_of_basis: int = 10,
        irreps_node_attr: int = "0e",
    ):

        assert pool_nodes is False

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.pool_nodes = pool_nodes
        self.irreps_node_attr = irreps_node_attr

        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        irreps_node_output = o3.Irreps(irreps_out)

        super().__init__()
        layers = []

        for _ in range(layers):
            layers.append(
                SingleMessagePassing(
                    irreps_node_input=irreps_in,
                    irreps_node_hidden=irreps_node_hidden,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_edge_attr=irreps_edge_attr,
                    fc_neurons=[self.number_of_basis, 100],
                    num_neighbors=self.num_neighbors,
                )
            )

            irreps_in = (
                layers[-1].mp.second.irreps_out + layers[-1].mp.second.irreps_out
            )

        self.layers = torch.nn.ModuleList(layers)
        self.final_layer = Convolution(
            irreps_in,
            self.irreps_node_attr,
            irreps_edge_attr,
            irreps_node_output,
            [self.number_of_basis, 100],
            num_neighbors,
        )

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Preprocesses the data for the forward pass.

        Args:
            data (Union[Data, Dict[str, torch.Tensor]]): Input data for the network.

        Returns:
            batch, x, edge_src, edge_dst, edge_vec: Processed data, including edge vectors
            for periodic boundary conditions.
        """

        return (
            data["batch"],
            data["x"],
            data["edge_src"],
            data["edge_dst"],
            data["edge_vec"],
        )

    def forward(self, data, aggregation_index):
        """
        Perform a forward pass of the model. with aggregation_index to compute 
        mean values over groups of nodes
        """
        _, node_inputs, edge_src, edge_dst, edge_vec = self.preprocess(data)
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

        for mp in self.layers:
            output = mp(
                node_inputs,
                node_attr,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedding,
            )
            mean_per_atom = torch_scatter.scatter_mean(
                output, aggregation_index, dim=0
            )[aggregation_index, :]

            node_inputs = torch.cat((output, mean_per_atom), dim=1)

        output = self.final_layer(
            node_inputs, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
        )
        print(output.shape)
        print(output)
        return output
