"""FP8 vs BF16 V-cache parity test. Pass kv dir + dtype as argv."""
import multiprocessing as mp
import os
import sys

os.environ["MASTER_PORT"] = "29506"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

KV_DIR = sys.argv[1]
V_DTYPE = sys.argv[2]  # "bf16" or "fp8"

if V_DTYPE == "fp8":
    os.environ["MSA_V_DTYPE"] = "fp8"

from memory_api import MemoryStore


def main():
    store = MemoryStore(kv_cache_dir=KV_DIR)

    store.add("菜花是一個 AI agent 住在 Docker 容器裡")
    store.add("RTX 3090 有 24GB VRAM")
    store.add("Yilan 是菜花的朋友 不是主人")
    store.add("CogVideoX-2b 品質差 PPT 等級")
    store.add("MSA 用 Sparse Transformer 做 retrieval 不是向量搜尋")

    print(f"[{V_DTYPE}] docs: {len(store)}")
    a1 = store.query("菜花是什麼？")
    print(f"[{V_DTYPE}] Q1: 菜花是什麼？")
    print(f"[{V_DTYPE}] A1: {a1}")
    a2 = store.query("RTX 3090 多少 VRAM？")
    print(f"[{V_DTYPE}] Q2: RTX 3090 多少 VRAM？")
    print(f"[{V_DTYPE}] A2: {a2}")
    a3 = store.query("MSA 跟向量搜尋有什麼不同？")
    print(f"[{V_DTYPE}] Q3: MSA 跟向量搜尋有什麼不同？")
    print(f"[{V_DTYPE}] A3: {a3}")

    store.close()
    print(f"[{V_DTYPE}] done")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
