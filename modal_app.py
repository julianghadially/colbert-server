"""Deploy colbert-server on Modal.

Usage:
    # Pre-download data into a persistent volume (~15 min, one-time)
    modal run modal_app.py::populate_volume

    # Deploy permanently (returns an HTTPS URL)
    modal deploy modal_app.py

    # Local dev with live reload
    modal serve modal_app.py
"""

from __future__ import annotations

from pathlib import Path

import modal

VOLUME_NAME = "colbert-wiki2017-data"
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

app = modal.App("colbert-server", image=image, volumes={DATA_DIR: volume})


@app.function(timeout=1800)
def populate_volume():
    """Pre-download the dataset into the persistent volume."""
    from colbert_server.data import detect_dataset_paths, download_collection_and_indexes

    print("Downloading collection and indexes …")
    snapshot_path = download_collection_and_indexes(cache_dir=HF_CACHE_DIR)
    print(f"Snapshot at {snapshot_path}")

    index_root, index_name, collection_path = detect_dataset_paths(snapshot_path)
    print(f"index_root={index_root}  index_name={index_name}  collection={collection_path}")

    DOWNLOAD_MARKER.touch()
    volume.commit()
    print("Volume committed. Data is ready.")


@app.cls(
    memory=16384,
    cpu=4.0,
    scaledown_window=3600,
    startup_timeout=900,
)
@modal.concurrent(max_inputs=100)
class ColbertService:
    @modal.enter()
    def load(self):
        from colbert_server.data import detect_dataset_paths, download_collection_and_indexes
        from colbert_server.server import create_searcher

        snapshot_path = download_collection_and_indexes(cache_dir=HF_CACHE_DIR)
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
