"""
Tests for building graphs using the neighbor list for atomic structures.

These tests compare graph-building functionality by using a class-based
approach for neighbor list construction and a function-based approach to
check if the generated edge sources and destinations match.
"""

import networkx as nx

from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader

from modules.property_prediction import RandomPropertiesPredictor
from modules.dataset import AtomsToGraphCollater


def test_graph_building(dataset):
    """
    Test the neighbor list construction for atomic structures with a
    specified cutoff of 5 using class-based neighbor list generation.

    This test compares the edge source and destination indices from the
    `get_cleaned_neighbours` function and the `ase.neighborlist.neighbor_list`
    function, ensuring that they are identical with skin = 1 and cutoff = 5
    """

    dataloader_func = DataLoader(dataset, batch_size=1, shuffle=False)
    dataloader_class = DataLoader(dataset, batch_size=1, shuffle=False)

    predictor = RandomPropertiesPredictor("cuda")

    dataloader_func.collate_fn = AtomsToGraphCollater(
        dataset=dataset,
        cutoff=5,
        noise_std=0,
        properties_predictor=predictor,
        forces_divided_by_mass=True,
        num_noisy_configurations=1,
        use_displacements=False,
        use_energies=False,
        upd_neigh_style="call_func",
        clip_value=0.01,
        strategy_sampling="trajectory",
        node_style_build="full_atoms",
        device="cuda",
    )

    dataloader_class.collate_fn = AtomsToGraphCollater(
        dataset=dataset,
        cutoff=5,
        noise_std=0,
        properties_predictor=predictor,
        forces_divided_by_mass=True,
        num_noisy_configurations=1,
        use_displacements=False,
        use_energies=False,
        upd_neigh_style="update_class",
        clip_value=0.01,
        strategy_sampling="trajectory",
        node_style_build="full_atoms",
        device="cuda",
    )

    for [data_class, _], [data_func, _] in zip(dataloader_func, dataloader_class):
        data_class = to_networkx(data_class.to_data_list()[0])
        data_func = to_networkx(data_func.to_data_list()[0])

        assert nx.is_isomorphic(data_class, data_func)
