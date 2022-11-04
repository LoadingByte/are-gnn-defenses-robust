import lzma
from io import BytesIO
from typing import Any, Optional, List, Set, Dict

import numpy as np
import seml
from gridfs import GridFS
from munch import Munch, munchify
from pymongo import MongoClient
from torchtyping import patch_typeguard
from tqdm.auto import tqdm
from typeguard import typechecked

patch_typeguard()

FALLBACK_SRC_PATH = "TODO"


@typechecked
def fetch(
        collection_name: str,
        fields: List[str],
        filter: Optional[Dict[str, Any]] = None,
        incl_files: Optional[Set[str]] = None
) -> List[Munch]:
    cfg = seml.database.get_mongodb_config()
    url = f"mongodb://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['db_name']}"
    filter = {**(filter or {}), "status": "COMPLETED"}
    with MongoClient(url) as client:
        db = client[cfg["db_name"]]
        fs = GridFS(db)
        coll = db[collection_name]
        exs = []
        for ex in tqdm(coll.find(filter, [*fields, "artifacts"]), total=coll.count_documents(filter), leave=False):
            ex = Munch(ex)  # not recursive
            ex.collection = collection_name
            if "config" in ex:
                ex.config = munchify(ex.config)  # recursive
            if "result" in ex:
                ex.result = Munch(ex.result)  # not recursive
                # Convert all python lists to numpy arrays to save memory.
                _lists_to_ndarrays(ex.result)
            # Merge all artifacts into the experiment dicts themselves.
            for arti in ex.pop("artifacts"):
                arti_name = arti["name"][:-7]  # strip the trailing .npz.xz
                if incl_files is not None and arti_name in incl_files:
                    with fs.get(arti["file_id"]) as grid_file:
                        xz_bytes = grid_file.read()
                    npz_bytes = lzma.decompress(xz_bytes)
                    for arr_name, arr in np.load(BytesIO(npz_bytes)).items():
                        parts = arr_name.split("/")
                        dct = ex.setdefault("result", {}).setdefault(arti_name, {})
                        for part in parts[:-1]:
                            dct = dct.setdefault(part, {})
                        dct[parts[-1]] = arr
            exs.append(ex)
        return exs


@typechecked
def _lists_to_ndarrays(dct: dict) -> None:
    for k, v in list(dct.items()):
        if isinstance(v, dict):
            _lists_to_ndarrays(v)
        elif isinstance(v, list):
            dct[k] = np.array(v)
