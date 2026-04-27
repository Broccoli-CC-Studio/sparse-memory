"""FP8 vs BF16 parity benchmark across many queries on a richer corpus.

Run twice: once with bf16, once with fp8, then diff the JSON outputs.

    uv run python test_fp8_multi.py kv_cache_multi_bf16 bf16 results_bf16.json
    uv run python test_fp8_multi.py kv_cache_multi_fp8  fp8  results_fp8.json
    uv run python compare_fp8_results.py results_bf16.json results_fp8.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

os.environ["MASTER_PORT"] = "29507"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

KV_DIR = sys.argv[1]
V_DTYPE = sys.argv[2]
OUT_PATH = sys.argv[3]

if V_DTYPE == "fp8":
    os.environ["MSA_V_DTYPE"] = "fp8"

from memory_api import MemoryStore


CORPUS = [
    "菜花是 Yilan 在 Docker 容器裡跑的 24/7 AI agent",
    "RTX 3090 顯卡有 24 GB VRAM",
    "MSA 全名 Memory Sparse Attention 是 Qwen3-4B 微調出來的稀疏 transformer",
    "MSA-4B model 在 HuggingFace EverMind-AI/MSA-4B 開源",
    "TCA-Attention 用 log-Gaussian 取樣產生 per-head sparsity budget",
    "DSV4 KV cache 用混合精度 BF16-RoPE + FP8-rest 達成 2x 壓縮",
    "DSV4 SWA branch n_win 設為 128 + attention sink",
    "InfMem 用 R_early = gamma 的 d-1 次方 reward shaping",
    "UMA 雙記憶結構 m_core summary slot + bank",
    "SCBench 包含 4 stage cache evaluation: generation / compression / retrieval / loading",
    "WUPHF 是 context management layer 跟 MSA 互補",
    "CogVideoX-2b 影片生成模型品質只有 PPT 等級",
    "Lenteja 是 character arena 的協作同事",
]

QUERIES = [
    "菜花住在哪裡？",
    "RTX 3090 多少 VRAM？",
    "MSA 是基於哪個基礎模型？",
    "MSA-4B 在哪個平台開源？",
    "TCA-Attention 怎麼產生 sparsity budget？",
    "DSV4 用什麼技術達成 KV cache 壓縮？",
    "DSV4 的 SWA branch n_win 是多少？",
    "InfMem 的 reward shaping 公式長什麼樣？",
    "UMA 的雙記憶結構是哪兩個部分？",
    "SCBench 評估幾個 cache stage？",
    "WUPHF 跟 MSA 是什麼關係？",
    "CogVideoX-2b 品質如何？",
    "Lenteja 是誰？",
]


def main():
    store = MemoryStore(kv_cache_dir=KV_DIR)
    for text in CORPUS:
        store.add(text)
    print(f"[{V_DTYPE}] corpus: {len(store)} docs")

    results = {"dtype": V_DTYPE, "corpus_size": len(store), "queries": []}

    for q in QUERIES:
        t0 = time.time()
        a = store.query(q)
        dt = time.time() - t0
        results["queries"].append({"q": q, "a": a, "latency_s": round(dt, 3)})
        print(f"[{V_DTYPE}] {dt:5.2f}s  Q: {q}")
        print(f"[{V_DTYPE}]        A: {a}")

    store.close()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{V_DTYPE}] saved {OUT_PATH}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
