"""Load test for the ColBERT search server.

Fires 50 distinct queries across 30 threads for 5 minutes, printing
live stats every 10 seconds and a final summary at the end.

Usage:
    python tests/load_test.py                           # default: http://localhost:8893
    python tests/load_test.py https://julianghadially--colbert-server-colbertservice-serve.modal.run    # custom URL
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier, Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

QUERIES = [
    "Who was the first president of the United States",
    "What is the speed of light in a vacuum",
    "How does photosynthesis work",
    "What causes earthquakes",
    "When was the internet invented",
    "What is the tallest mountain in the world",
    "How do vaccines work",
    "What is dark matter",
    "Who wrote Romeo and Juliet",
    "What is the theory of relativity",
    "How far is the moon from Earth",
    "What is machine learning",
    "When did World War II end",
    "How does a nuclear reactor work",
    "What is the largest ocean on Earth",
    "Who discovered penicillin",
    "What is the greenhouse effect",
    "How do black holes form",
    "What is quantum computing",
    "When was the printing press invented",
    "What is the human genome project",
    "How does GPS work",
    "What causes the northern lights",
    "Who painted the Mona Lisa",
    "What is CRISPR gene editing",
    "How do solar panels generate electricity",
    "What is the Big Bang theory",
    "When did humans first land on the moon",
    "What is blockchain technology",
    "How does the immune system work",
    "What is the deepest point in the ocean",
    "Who invented the telephone",
    "What is artificial intelligence",
    "How do airplanes fly",
    "What causes climate change",
    "When was the first computer built",
    "What is the periodic table of elements",
    "How does DNA replication work",
    "What is the theory of evolution",
    "Who discovered gravity",
    "What is superconductivity",
    "How do earthquakes cause tsunamis",
    "What is the speed of sound",
    "When was electricity discovered",
    "What is the Higgs boson",
    "How does radar work",
    "What are stem cells",
    "Who built the Great Wall of China",
    "What is nuclear fusion",
    "How does the stock market work",
]

K = 50

THREADS = 30
DURATION_SECONDS = 5 * 60


def send_query(base_url: str, query: str) -> tuple[int, float]:
    """Send a single search request. Returns (status_code, latency_seconds)."""
    params = urlencode({"query": query, "k": K})
    url = f"{base_url}/api/search?{params}"
    req = Request(url)
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as resp:
            resp.read()
            return resp.status, time.perf_counter() - start
    except HTTPError as e:
        return e.code, time.perf_counter() - start
    except (URLError, TimeoutError):
        return 0, time.perf_counter() - start


def worker(base_url: str, stats: dict, lock: Lock, barrier: Barrier, deadline: float):
    """Wait for all threads to be ready, then blast queries until the deadline."""
    barrier.wait()
    while time.monotonic() < deadline:
        query = random.choice(QUERIES)
        status, latency = send_query(base_url, query)

        with lock:
            stats["total"] += 1
            stats["latencies"].append(latency)
            if 200 <= status < 300:
                stats["success"] += 1
            elif status == 0:
                stats["timeouts"] += 1
            else:
                stats["errors"] += 1


def print_stats(stats: dict, lock: Lock, elapsed: float):
    with lock:
        total = stats["total"]
        success = stats["success"]
        errors = stats["errors"]
        timeouts = stats["timeouts"]
        latencies = stats["latencies"]

    if not latencies:
        print(f"  {elapsed:6.0f}s | no requests completed yet")
        return

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else p50
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 2 else p50
    rps = total / elapsed if elapsed > 0 else 0

    print(
        f"  {elapsed:6.0f}s | "
        f"reqs: {total:>6}  ok: {success:>6}  err: {errors:>3}  timeout: {timeouts:>3} | "
        f"rps: {rps:6.1f} | "
        f"p50: {p50:.3f}s  p95: {p95:.3f}s  p99: {p99:.3f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="ColBERT server load test")
    parser.add_argument("url", nargs="?", default="http://localhost:8893", help="Base URL of the server")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(f"Load test: {len(QUERIES)} queries, {THREADS} threads, {DURATION_SECONDS}s, k={K}")
    print(f"Target: {base_url}")
    print("-" * 90)

    stats = {"total": 0, "success": 0, "errors": 0, "timeouts": 0, "latencies": []}
    lock = Lock()
    barrier = Barrier(THREADS)
    deadline = time.monotonic() + DURATION_SECONDS
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(worker, base_url, stats, lock, barrier, deadline) for _ in range(THREADS)]

        while time.monotonic() < deadline:
            time.sleep(10)
            print_stats(stats, lock, time.monotonic() - start_time)

        for f in as_completed(futures):
            f.result()

    elapsed = time.monotonic() - start_time
    print("-" * 90)
    print("FINAL RESULTS")
    print("-" * 90)
    print_stats(stats, lock, elapsed)

    with lock:
        latencies = stats["latencies"]
    if latencies:
        print(f"\n  min: {min(latencies):.3f}s  max: {max(latencies):.3f}s  "
              f"mean: {statistics.mean(latencies):.3f}s  stdev: {statistics.stdev(latencies):.3f}s"
              if len(latencies) >= 2
              else f"\n  min: {min(latencies):.3f}s  max: {max(latencies):.3f}s")

    error_rate = (stats["errors"] + stats["timeouts"]) / max(stats["total"], 1) * 100
    print(f"\n  error rate: {error_rate:.1f}%")
    sys.exit(0 if error_rate < 5 else 1)


if __name__ == "__main__":
    main()
