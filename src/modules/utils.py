"""
This module contains utility functions for caching and querying materials from 
the Materials Project API.

Functions:
- setup_cache: Set up a cache using joblib.
- _query_mpid_structure: Query the Materials Project database for structures based on MPIDs.
- query_mpid_structure: Public function to query material structures based on MPIDs.
"""

import os
from functools import lru_cache
from typing import List, Optional, Union

import joblib
from mp_api.client import MPRester


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
