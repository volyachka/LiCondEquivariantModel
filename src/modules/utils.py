"""
This module contains utility functions
"""

import os
from functools import lru_cache
from typing import List, Optional, Union

import joblib
from mp_api.client import MPRester

import numpy as np
from ase.neighborlist import NeighborList
from pymatgen.io.ase import MSONAtoms


CACHE_ENV_VAR = "PFP_CACHE"
DEFAULT_CACHE_PATH = "./cache"


@lru_cache
def setup_cache(cache_path: Optional[str] = None) -> joblib.Memory:
    """
    Set up a cache using joblib's Memory caching system.

    Args:
        cache_path: Path to the cache directory. If not provided, it uses the
                    `PFP_CACHE` environment variable or defaults to `./cache`.

    Returns:
        A joblib.Memory instance for caching.
    """
    if cache_path is None:
        cache_path = os.getenv(CACHE_ENV_VAR, DEFAULT_CACHE_PATH)
        print(
            f'Cache path set to "{cache_path}". '
            f"To change, set {CACHE_ENV_VAR} environment variable."
        )

    return joblib.Memory(cache_path)


_memory = setup_cache()


@_memory.cache
def _query_mpid_structure(mpids):
    """
    Query the Materials Project database for structure data using the given MPIDs.

    Args:
        mpids: List of MPIDs or a single MPID to query for.

    Returns:
        A list of dictionaries containing structure data for each MPID.
    """
    with open(os.path.abspath(".mp_apikey"), encoding="utf-8") as f:
        mp_api_key = f.read().strip()

    with MPRester(mp_api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=mpids, fields=["structure", "material_id"]
        )
    return [d.model_dump() for d in docs]


def query_mpid_structure(mpids: Union[List[str], str]) -> List[dict]:
    """
    Query the Materials Project database for structure data based on MPIDs.

    Args:
        mpids: A list or single string representing MPIDs to query for.

    Returns:
        A list of dictionaries containing structure data for each
        queried MPID, sorted by material ID.
    """
    if isinstance(mpids, str):
        mpids = [mpids]

    return sorted(
        _query_mpid_structure(sorted(mpids)),
        key=lambda doc: int(doc["material_id"][3:]),
    )


def get_cleaned_neighbours_li_grouped(  # pylint: disable=R0914
    nl_builder: NeighborList,
    noise_structure: MSONAtoms,
    cutoff: float,
):
    """
    Get the cleaned neighbor list from a noisy atomic structure.

    This function builds a neighbor list for a given noisy atomic structure,
    filters the neighbors based on the cutoff distance, and returns the
    corresponding edge indices and vectors. The atomic positions are updated
    using the neighbor list builder, and the distances are adjusted according
    to the lattice for periodic systems.
    """

    pos = noise_structure.get_positions()
    mask = np.full(pos.shape[0], False, dtype=bool)
    lattice = noise_structure.cell.array

    nl_builder.update(noise_structure)

    src = []
    dst = []
    shifts = []

    li_symbols = noise_structure.symbols == "Li"

    for i_atom, name in enumerate(noise_structure.symbols):

        neighbor_ids, offsets = nl_builder.get_neighbors(i_atom)
        rr = pos[neighbor_ids] + offsets @ lattice - pos[i_atom][None, :]

        if cutoff is not None:
            suitable_neigh = np.linalg.norm(rr, axis=1) <= cutoff
            neighbor_ids = neighbor_ids[suitable_neigh]
            offsets = offsets[suitable_neigh]
            rr = rr[suitable_neigh]

        if name != "Li":
            suitable_neigh = li_symbols[neighbor_ids]
            neighbor_ids = neighbor_ids[suitable_neigh]
            offsets = offsets[suitable_neigh]
            rr = rr[suitable_neigh]

        mask[i_atom] = True
        mask[neighbor_ids] = True

        src.append(np.ones(len(neighbor_ids)) * i_atom)
        dst.append(neighbor_ids)
        shifts.append(offsets)

    edge_dst = np.concatenate(dst).astype(np.int64)
    edge_src = np.concatenate(src).astype(np.int64)
    shifts = np.concatenate(shifts).astype(np.int64)

    return edge_src, edge_dst, shifts, mask


def get_cleaned_neighbours(
    nl_builder: NeighborList, noise_structure: MSONAtoms, cutoff: float
):
    """
    Get the cleaned neighbor list from a noisy atomic structure.

    This function builds a neighbor list for a given noisy atomic structure,
    filters the neighbors based on the cutoff distance, and returns the
    corresponding edge indices and vectors. The atomic positions are updated
    using the neighbor list builder, and the distances are adjusted according
    to the lattice for periodic systems.
    """

    pos = noise_structure.get_positions()
    lattice = noise_structure.cell.array

    nl_builder.update(noise_structure)

    src = []
    dst = []
    shifts = []

    for i_atom, _ in enumerate(noise_structure.symbols):
        neighbor_ids, offsets = nl_builder.get_neighbors(i_atom)
        rr = pos[neighbor_ids] + offsets @ lattice - pos[i_atom][None, :]

        if cutoff is not None:
            suitable_neigh = np.linalg.norm(rr, axis=1) <= cutoff
            neighbor_ids = neighbor_ids[suitable_neigh]
            offsets = offsets[suitable_neigh]
            rr = rr[suitable_neigh]

        src.append(np.ones(len(neighbor_ids)) * i_atom)
        dst.append(neighbor_ids)
        shifts.append(offsets)

    edge_dst = np.concatenate(dst).astype(np.int64)
    edge_src = np.concatenate(src).astype(np.int64)
    shifts = np.concatenate(shifts).astype(np.int64)
    return edge_src, edge_dst, shifts
