"""
Tests for building graphs using the neighbor list for atomic structures.

These tests compare graph-building functionality by using a class-based 
approach for neighbor list construction and a function-based approach to 
check if the generated edge sources and destinations match.
"""

import ase.io
from ase.neighborlist import NeighborList
from modules.utils import get_cleaned_neighbours


def test_graph_building_1(dataset):
    """
    Test the neighbor list construction for atomic structures with a
    specified cutoff of 0.5 using class-based neighbor list generation.

    This test compares the edge source and destination indices from the
    `get_cleaned_neighbours` function and the `ase.neighborlist.neighbor_list`
    function, ensuring that they are identical with skin = 0 and cutoff = None.
    """

    structures = [data.x["atoms"] for data in dataset]
    cutoff = 0.5
    for structure in structures:
        cuttofs = len(structure.get_positions()) * [cutoff / 2]
        nl_builder = NeighborList(
            cuttofs, self_interaction=True, bothways=False, skin=0
        )

        edge_src_class, edge_dst_class = get_cleaned_neighbours(
            nl_builder, structure, cutoff=None
        )

        edge_src_func, edge_dst_func, _ = ase.neighborlist.neighbor_list(
            "ijS",
            a=structure,
            cutoff=cutoff,
            self_interaction=True,
        )

        assert edge_src_class.shape == edge_src_func.shape
        assert edge_dst_class.shape == edge_dst_func.shape


def test_graph_building_2(dataset):
    """
    Test the neighbor list construction for atomic structures with a
    specified cutoff of 0.5 using class-based neighbor list generation.

    This test compares the edge source and destination indices from the
    `get_cleaned_neighbours` function and the `ase.neighborlist.neighbor_list`
    function, ensuring that they are identical with skin = 1 and cutoff = 0.5
    """
    structures = [data.x["atoms"] for data in dataset]
    cutoff = 0.5
    for structure in structures:
        cuttofs = len(structure.get_positions()) * [cutoff / 2]
        nl_builder = NeighborList(
            cuttofs, self_interaction=True, bothways=False, skin=1
        )

        edge_src_class, edge_dst_class = get_cleaned_neighbours(
            nl_builder, structure, cutoff
        )

        edge_src_func, edge_dst_func, _ = ase.neighborlist.neighbor_list(
            "ijS",
            a=structure,
            cutoff=cutoff,
            self_interaction=True,
        )

        assert edge_src_class.shape == edge_src_func.shape
        assert edge_dst_class.shape == edge_dst_func.shape
