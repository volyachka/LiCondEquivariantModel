"""
Mixing Network Module

This module contains the implementation of a mixing network using message passing layers.
"""

from typing import Dict, Union

import torch
from torch_geometric.data import Data
import torch_scatter
from e3nn import o3
from e3nn.nn import Gate


from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2103.gate_points_message_passing import (
    Convolution,
    Compose,
    tp_path_exists,
)


# pylint: disable=R0913, R0914, R0917, R0801
class SingleMessagePassing(torch.nn.Module):
    r"""

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input features

    irreps_node_hidden : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self,
        irreps_node_input: str,
        irreps_node_hidden: str,
        irreps_node_attr: str,
        irreps_edge_attr: str,
        fc_neurons: int,
        num_neighbors: int,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        irreps_node = self.irreps_node_input

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node_hidden
                if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ]
        ).simplify()

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ]
        )
        ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

        gate = Gate(
            irreps_scalars,
            [act[ir.p] for _, ir in irreps_scalars],  # scalar
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated,  # gated tensors
        )

        conv = Convolution(
            irreps_node,
            self.irreps_node_attr,
            self.irreps_edge_attr,
            gate.irreps_in,
            fc_neurons,
            num_neighbors,
        )
        irreps_node = gate.irreps_out
        self.mp = Compose(conv, gate)

    def forward(
        self,
        node_features: torch.Tensor,
        node_attr: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the message passing layer."""

        node_features = self.mp(
            node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
        )

        return node_features


# pylint: disable=R0913, R0902, R0917
class MixingNetwork(torch.nn.Module):
    """
    A network that adapts the SimpleNetwork functional with possibility to share
    features between message passing iterations
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
        irreps_node_attr: int = "0e",
    ):

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.irreps_node_attr = irreps_node_attr
        self.num_neighbors = num_neighbors
        assert pool_nodes is False
        assert self.num_nodes == 1
        assert fc_neurons[0] == self.number_of_basis
        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )

        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        irreps_node_output = o3.Irreps(irreps_out)

        super().__init__()
        mp_layers = []

        for _ in range(layers):
            mp_layers.append(
                SingleMessagePassing(
                    irreps_node_input=irreps_in,
                    irreps_node_hidden=irreps_node_hidden,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_edge_attr=irreps_edge_attr,
                    fc_neurons=fc_neurons,
                    num_neighbors=self.num_neighbors,
                )
            )

            irreps_in = (
                mp_layers[-1].mp.second.irreps_out + mp_layers[-1].mp.second.irreps_out
            )

        self.mp_layers = torch.nn.ModuleList(mp_layers)
        self.final_layer = Convolution(
            irreps_in,
            self.irreps_node_attr,
            irreps_edge_attr,
            irreps_node_output,
            fc_neurons=fc_neurons,
            num_neighbors=num_neighbors,
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

        batch = data["batch"]

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

    def forward(
        self,
        data: Union[Data, Dict[str, torch.Tensor]],
        aggregation_index: torch.tensor,
    ) -> torch.tensor:
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

        for mp in self.mp_layers:
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
        return output
