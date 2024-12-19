from mp_api.client import MPRester
import os
from functools import lru_cache
import joblib
from typing import Dict, List, Optional, Union
from tqdm import tqdm

CACHE_ENV_VAR = "PFP_CACHE"
DEFAULT_CACHE_PATH = "./cache"


@lru_cache
def setup_cache(
    cache_path: Optional[str] = None,
) -> joblib.Memory:
    if cache_path is None:
        cache_path = os.getenv(CACHE_ENV_VAR, DEFAULT_CACHE_PATH)
        print(
            f'Cache path set to "{cache_path}". To change, set {CACHE_ENV_VAR} environment variable.'
        )

    return joblib.Memory(cache_path)


_memory = setup_cache()


@_memory.cache
def _query_mpid_structure(mpids):
    with open(os.path.abspath(".mp_apikey")) as f:
        MP_API_KEY = f.read().strip()

    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=mpids, fields=["structure", "material_id"]
        )
    return [d.model_dump() for d in docs]


def query_mpid_structure(mpids: Union[List[str], str]) -> List[dict]:
    if isinstance(mpids, str):
        mpids = [mpids]

    return sorted(
        _query_mpid_structure(sorted(mpids)),
        key=lambda doc: int(doc["material_id"][3:]),
    )
