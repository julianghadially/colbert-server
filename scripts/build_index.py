"""Build a ColBERT index from a Phantom Wiki articles.json file.

Usage:
    python scripts/build_index.py /path/to/articles.json /path/to/output_dir

    uv run python scripts/build_index.py \
    /Users/julianghadially/Documents/code/phantom-wiki/output/depth_10_size_1000000/articles.json \
    database/phantom-wiki

This will:
1. Convert articles.json → collection.tsv (one passage per row)
2. Build a ColBERT index using colbertv2.0 checkpoint

The output directory will contain:
    collection/collection.tsv   - the text passages
    indexes/<index_name>/       - the ColBERT index files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def articles_to_collection(articles_path: str, output_dir: str) -> str:
    """Convert articles.json to collection.tsv for ColBERT."""
    collection_dir = os.path.join(output_dir, "collection")
    os.makedirs(collection_dir, exist_ok=True)
    collection_path = os.path.join(collection_dir, "collection.tsv")

    print(f"Reading {articles_path}...")
    with open(articles_path) as f:
        articles = json.load(f)

    print(f"Writing {len(articles)} passages to {collection_path}...")
    with open(collection_path, "w") as f:
        for pid, article in enumerate(articles):
            # Combine title and article text as the passage
            # Replace tabs and newlines to keep TSV format intact
            title = article["title"].replace("\t", " ").replace("\n", " ")
            text = article["article"].replace("\t", " ").replace("\n", " ")
            passage = f"{title}: {text}"
            f.write(f"{pid}\t{passage}\n")

    print(f"Created collection with {len(articles)} passages.")
    return collection_path


def build_index(
    collection_path: str,
    output_dir: str,
    index_name: str = "phantom-wiki",
    checkpoint: str = "colbert-ir/colbertv2.0",
    nbits: int = 2,
) -> None:
    """Build a ColBERT index from a collection.tsv file."""
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    index_root = os.path.join(output_dir, "indexes")
    os.makedirs(index_root, exist_ok=True)

    print(f"Building ColBERT index '{index_name}' with {nbits}-bit quantization...")
    print(f"Checkpoint: {checkpoint}")
    print(f"Collection: {collection_path}")
    print(f"Index root: {index_root}")
    print("This may take a while for large collections...")

    with Run().context(RunConfig(nranks=1, root=output_dir)):
        config = ColBERTConfig(
            doc_maxlen=512,
            nbits=nbits,
            root=output_dir,
            index_root=index_root,
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection_path, overwrite=True)

    print(f"Index built at {index_root}/{index_name}")


def main():
    parser = argparse.ArgumentParser(description="Build ColBERT index from articles.json")
    parser.add_argument("articles_path", help="Path to articles.json")
    parser.add_argument("output_dir", help="Directory to write collection and index")
    parser.add_argument("--index-name", default="phantom-wiki", help="Name for the index")
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0", help="ColBERT checkpoint")
    parser.add_argument("--nbits", type=int, default=2, help="Quantization bits (default: 2)")
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Only create collection.tsv, skip index building",
    )
    args = parser.parse_args()

    if not os.path.exists(args.articles_path):
        print(f"Error: {args.articles_path} not found")
        sys.exit(1)

    collection_path = articles_to_collection(args.articles_path, args.output_dir)

    if args.collection_only:
        print("Skipping index build (--collection-only).")
        print(f"\nTo build the index later, run:")
        print(f"  python scripts/build_index.py {args.articles_path} {args.output_dir}")
        return

    build_index(
        collection_path,
        args.output_dir,
        index_name=args.index_name,
        checkpoint=args.checkpoint,
        nbits=args.nbits,
    )

    print(f"\nDone! To serve this index:")
    index_root = os.path.join(args.output_dir, "indexes")
    print(
        f"  colbert-server serve "
        f"--index-root {index_root} "
        f"--index-name {args.index_name} "
        f"--collection-path {collection_path}"
    )


if __name__ == "__main__":
    main()
