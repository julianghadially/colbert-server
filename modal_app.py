"""Deploy colbert-server on Modal.

Usage:
    # Pre-download data into a persistent volume (~15 min, one-time)
    modal run modal_app.py::populate_volume

    # Deploy permanently (returns an HTTPS URL)
    modal deploy modal_app.py

    # Local dev with live reload
    modal serve modal_app.py

Select the dataset via the DATASET env var (default: wiki):
    DATASET=phantom-wiki modal run modal_app.py::populate_volume
    DATASET=phantom-wiki modal deploy modal_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

DATASETS = {
    "wiki": {
        "volume_name": "colbert-wiki2017-data",
        "repo_id": "nielsgl/colbert-wiki2017",
    },
    "phantom-wiki": {
        "volume_name": "colbert-phantom-wiki-data",
        "repo_id": "julianghadially/phantom-wiki-colbert-index",
    },
}

DATASET = os.environ.get("DATASET", "wiki")
if DATASET not in DATASETS:
    raise ValueError(f"Unknown DATASET={DATASET!r}. Choose from: {', '.join(DATASETS)}")

_config = DATASETS[DATASET]
VOLUME_NAME = _config["volume_name"]
REPO_ID = _config["repo_id"]

DATA_DIR = "/data"
HF_CACHE_DIR = Path(DATA_DIR) / "hf_cache"
DOWNLOAD_MARKER = Path(DATA_DIR) / ".download_complete"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        "torch>=2.9.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "colbert-ai>=0.2.22",
        "faiss-cpu>=1.12.0",
        "flask>=3.1.2",
        "huggingface-hub>=0.36.0",
        "packaging>=24.1",
        "transformers==4.57.1",
    )
    .add_local_python_source("colbert_server")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(f"colbert-server-{DATASET}", image=image, volumes={DATA_DIR: volume})


@app.function(timeout=1800)
def populate_volume():
    """Pre-download the dataset into the persistent volume."""
    from colbert_server.data import detect_dataset_paths, download_collection_and_indexes

    print(f"Downloading collection and indexes for {DATASET!r} ({REPO_ID}) …")
    snapshot_path = download_collection_and_indexes(
        repo_id=REPO_ID, cache_dir=HF_CACHE_DIR
    )
    print(f"Snapshot at {snapshot_path}")

    index_root, index_name, collection_path = detect_dataset_paths(snapshot_path)
    print(f"index_root={index_root}  index_name={index_name}  collection={collection_path}")

    DOWNLOAD_MARKER.touch()
    volume.commit()
    print("Volume committed. Data is ready.")


@app.cls(
    memory=16384,
    cpu=4.0,
    min_containers=1,
    max_containers=5,
    scaledown_window=3600,
    startup_timeout=900,
)
@modal.concurrent(max_inputs=6)
class ColbertService:
    @modal.enter()
    def load(self):
        from colbert_server.data import detect_dataset_paths, download_collection_and_indexes
        from colbert_server.server import create_searcher

        snapshot_path = download_collection_and_indexes(
            repo_id=REPO_ID, cache_dir=HF_CACHE_DIR
        )
        index_root, index_name, collection_path = detect_dataset_paths(snapshot_path)

        self.searcher = create_searcher(
            index_root=str(index_root),
            index_name=index_name,
            collection_path=str(collection_path) if collection_path else None,
        )

    @modal.wsgi_app()
    def serve(self):
        from colbert_server.server import create_app

        return create_app(self.searcher)
