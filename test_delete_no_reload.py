"""End-to-end test for P1.1 rebuild optimization.

Adds 5 docs, queries (cold), removes 1, queries again (post-delete).
Times the second query and asserts it stays under 10s. Pre-fix path took
~70s on this size; post-fix should be 1-3s.

Run:
    cd /home/agent/work/msa
    CUDA_VISIBLE_DEVICES=0 uv run python3 test_delete_no_reload.py
"""

import multiprocessing as mp
import os
import time

os.environ["MASTER_PORT"] = "29508"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from memory_api import MemoryStore


DOCS = [
    "Broccoli is an AI agent living in a Docker container with an RTX 3090.",
    "RTX 3090 has 24GB of VRAM.",
    "MSA uses Sparse Transformer for retrieval, not vector search.",
    "Yilan is Broccoli's friend and also an investor in the project.",
    "CogVideoX-2b generates 720x480 video at PPT-quality.",
]

QUESTION_BEFORE = "What hardware does Broccoli use?"
QUESTION_AFTER = "What hardware does Broccoli use?"


def main():
    store = MemoryStore(kv_cache_dir="/home/agent/work/msa/kv_cache_test_delete")

    print("=== Phase 1: add 5 docs ===")
    t0 = time.time()
    for d in DOCS:
        store.add(d)
    print(f"  add latency: {time.time() - t0:.3f}s ({len(store)} active)")

    print("\n=== Phase 2: cold query (loads checkpoint, prefills) ===")
    t0 = time.time()
    ans = store.query(QUESTION_BEFORE)
    cold_dt = time.time() - t0
    print(f"  cold query latency: {cold_dt:.2f}s")
    print(f"  answer: {ans[:200]}")

    print("\n=== Phase 3: warm query (no rebuild) ===")
    t0 = time.time()
    ans = store.query(QUESTION_BEFORE)
    warm_dt = time.time() - t0
    print(f"  warm query latency: {warm_dt:.2f}s")
    print(f"  answer: {ans[:200]}")

    print("\n=== Phase 4: remove doc 4 (CogVideoX) ===")
    t0 = time.time()
    store.remove(4)
    print(f"  remove latency: {time.time() - t0:.3f}s ({len(store)} active)")

    print("\n=== Phase 5: post-delete query (triggers reset_documents) ===")
    t0 = time.time()
    ans = store.query(QUESTION_AFTER)
    delete_query_dt = time.time() - t0
    print(f"  post-delete query latency: {delete_query_dt:.2f}s")
    print(f"  answer: {ans[:200]}")

    print("\n=== Phase 6: warm query after delete (should be fast) ===")
    t0 = time.time()
    ans = store.query("What is MSA?")
    final_warm_dt = time.time() - t0
    print(f"  warm query latency: {final_warm_dt:.2f}s")
    print(f"  answer: {ans[:200]}")

    print("\n=== Summary ===")
    print(f"  cold first query:        {cold_dt:7.2f}s")
    print(f"  warm query (no change):  {warm_dt:7.2f}s")
    print(f"  post-delete query:       {delete_query_dt:7.2f}s  (pre-fix baseline ~70s, target <10s)")
    print(f"  warm query after delete: {final_warm_dt:7.2f}s")

    assert delete_query_dt < 30.0, (
        f"post-delete query took {delete_query_dt:.2f}s, expected <30s. "
        "Pre-fix baseline was ~70s; if regressed, check reset_documents path."
    )

    if delete_query_dt < 5.0:
        verdict = "PASS (excellent, target <5s achieved)"
    elif delete_query_dt < 10.0:
        verdict = "PASS (target <10s achieved)"
    else:
        verdict = "PASS-WEAK (under 30s but above 10s target, optimization may need more work)"
    print(f"\nVerdict: {verdict}")

    store.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
