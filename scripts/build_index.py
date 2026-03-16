"""Build a ColBERT index from a Phantom Wiki dataset.

Usage:
    # From a local articles.json file:
    python scripts/build_index.py /path/to/articles.json database/phantom-wiki

    # From a Hugging Face dataset:
    python scripts/build_index.py --from-hf julianghadially/phantom-wiki-depth10-1M-seed45 database/phantom-wiki

This will:
1. Convert articles → collection.tsv (one passage per row)
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


def hf_dataset_to_collection(repo_id: str, output_dir: str) -> str:
    """Download a Hugging Face dataset and convert to collection.tsv for ColBERT."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required for --from-hf.")
        print("Install it with: uv pip install datasets")
        sys.exit(1)

    collection_dir = os.path.join(output_dir, "collection")
    os.makedirs(collection_dir, exist_ok=True)
    collection_path = os.path.join(collection_dir, "collection.tsv")

    print(f"Downloading dataset from Hugging Face: {repo_id}...")
    dataset = load_dataset(repo_id, split="train")

    # Save raw articles as articles.json so the repo has the source data
    articles_path = os.path.join(output_dir, "articles.json")
    print(f"Saving {len(dataset)} articles to {articles_path}...")
    articles = [{"title": row["title"], "article": row["article"]} for row in dataset]
    with open(articles_path, "w") as f:
        json.dump(articles, f)
    print(f"Saved articles.json ({len(articles)} articles).")

    print(f"Writing {len(dataset)} passages to {collection_path}...")
    with open(collection_path, "w") as f:
        for pid, article in enumerate(articles):
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
    parser = argparse.ArgumentParser(description="Build ColBERT index from articles")
    parser.add_argument(
        "articles_path",
        nargs="?",
        default=None,
        help="Path to articles.json (not needed when using --from-hf)",
    )
    parser.add_argument("output_dir", help="Directory to write collection and index")
    parser.add_argument(
        "--from-hf",
        metavar="REPO_ID",
        help="Download dataset from Hugging Face instead of a local file "
        "(e.g. julianghadially/phantom-wiki-depth10-1M-seed45)",
    )
    parser.add_argument("--index-name", default="phantom-wiki", help="Name for the index")
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0", help="ColBERT checkpoint")
    parser.add_argument("--nbits", type=int, default=2, help="Quantization bits (default: 2)")
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Only create collection.tsv, skip index building",
    )
    args = parser.parse_args()

    if args.from_hf:
        collection_path = hf_dataset_to_collection(args.from_hf, args.output_dir)
    elif args.articles_path:
        if not os.path.exists(args.articles_path):
            print(f"Error: {args.articles_path} not found")
            sys.exit(1)
        collection_path = articles_to_collection(args.articles_path, args.output_dir)
    else:
        parser.error("Either provide articles_path or use --from-hf REPO_ID")

    if args.collection_only:
        print("Skipping index build (--collection-only).")
        print(f"\nTo build the index later, re-run without --collection-only.")
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
