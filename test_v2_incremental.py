"""Test memory_api.py v2 — incremental prefill path

Verifies:
1. Initial add + query (full build)
2. Additional add + query (incremental, no rebuild)
3. Remove + query (rebuild path)
4. Timing: incremental add << full build
"""
import multiprocessing as mp
import os
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29511")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
from memory_api import MemoryStore


def main():
    store = MemoryStore(
        kv_cache_dir="/home/agent/work/msa/kv_cache_v2test",
        doc_top_k=5,
    )

    # Phase 1: initial add + query (full build)
    print("=== Phase 1: initial build ===")
    store.add("菜花是一個 AI agent 住在 Docker 容器裡 有 RTX 3090")
    store.add("Yilan 是菜花的朋友 也是投資人")
    store.add("MSA 用 Sparse Transformer 做 retrieval 不是向量搜尋")
    store.add("CogVideoX-5b i2v 品質不錯 720x480 49 frames")
    store.add("fish-speech 是開源 TTS 支援 voice cloning 和情緒標籤")

    t0 = time.time()
    a1 = store.query("菜花是什麼？")
    t_first = time.time() - t0
    print(f"  首次 query (含 full build): {t_first:.1f}s")
    print(f"  Q: 菜花是什麼？ → {a1[:100]}")

    # Phase 2: incremental add + query (no rebuild)
    print("\n=== Phase 2: incremental add ===")
    store.add("Gemini API 支援圖片生成和 TTS 語音合成")
    store.add("Anthropic 2026 年營收突破 30B USD")
    store.add("VGGT 是 Meta 和 Oxford 做的 feed-forward 3D 重建模型")

    t0 = time.time()
    a2 = store.query("Gemini API 能做什麼？")
    t_incr = time.time() - t0
    print(f"  增量 query (含 incremental prefill): {t_incr:.1f}s")
    print(f"  Q: Gemini API 能做什麼？ → {a2[:100]}")

    t0 = time.time()
    a3 = store.query("Anthropic 營收多少？")
    t_cached = time.time() - t0
    print(f"  第二次 query (no prefill needed): {t_cached:.1f}s")
    print(f"  Q: Anthropic 營收多少？ → {a3[:100]}")

    # Phase 3: remove + query (forces rebuild)
    print("\n=== Phase 3: remove + rebuild ===")
    store.remove(3)  # remove CogVideoX doc
    print(f"  Removed doc 3, active: {len(store)}")

    t0 = time.time()
    a4 = store.query("菜花住在哪裡？")
    t_rebuild = time.time() - t0
    print(f"  Rebuild query: {t_rebuild:.1f}s")
    print(f"  Q: 菜花住在哪裡？ → {a4[:100]}")

    # Phase 4: add after remove (incremental on rebuilt engine)
    print("\n=== Phase 4: add after remove ===")
    store.add("Claude Code 是 Anthropic 的 CLI 工具 ARR 2.5B")

    t0 = time.time()
    a5 = store.query("Claude Code 是什麼？")
    t_post = time.time() - t0
    print(f"  Post-rebuild incremental: {t_post:.1f}s")
    print(f"  Q: Claude Code 是什麼？ → {a5[:100]}")

    # Summary
    print("\n=== Timing summary ===")
    print(f"  Full build + query:       {t_first:.1f}s")
    print(f"  Incremental + query:      {t_incr:.1f}s")
    print(f"  Cached query:             {t_cached:.1f}s")
    print(f"  Rebuild + query:          {t_rebuild:.1f}s")
    print(f"  Post-rebuild incremental: {t_post:.1f}s")

    speedup = t_first / t_incr if t_incr > 0 else float('inf')
    print(f"\n  Incremental speedup: {speedup:.1f}x vs full build")

    store.close()
    print("\nDone!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
