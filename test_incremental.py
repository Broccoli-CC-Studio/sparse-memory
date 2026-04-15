"""Test incremental prefill — add documents without restarting engine"""
import multiprocessing as mp
import os
os.environ["MASTER_PORT"] = "29507"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.msa_service import MSAEngine
from src.config.memory_config import GenerateConfig, ModelConfig, MemoryConfig
import json
import time

MODEL_PATH = "ckpt/MSA-4B"
INITIAL_DOCS = "test_memory.json"  # 10 docs

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    cfg = json.load(f)
msa_cfg = cfg.get("msa_config", {})

generate_config = GenerateConfig(
    devices=[0],
    template="QWEN3_INSTRUCT_TEMPLATE",
    max_generate_tokens=256,
    top_p=0.9,
    temperature=0.0,
    qa_mode=True,
)
model_config = ModelConfig(
    model_path=MODEL_PATH,
    doc_top_k=5,
    pooling_kernel_size=msa_cfg.get("pooling_kernel_size", 64),
    router_layer_idx=msa_cfg.get("router_layer_idx", "all"),
)
memory_config = MemoryConfig(
    block_size=16000,
    pooling_kernel_size=msa_cfg.get("pooling_kernel_size", 64),
    slice_chunk_size=16 * 1024,
    memory_file_path=INITIAL_DOCS,
)

def main():
    with MSAEngine(generate_config, model_config, memory_config) as engine:
        idx_to_doc = engine.get_idx_to_doc()
        print(f"Initial: {len(idx_to_doc)} docs")

        # Query before adding
        print("\n--- Before add_documents ---")
        texts, _, _ = engine.generate("Gemini API 能做什麼？", require_recall_topk=True)
        answer = texts[0]
        marker = "The answer to the question is:"
        if marker in answer:
            print(f"Q: Gemini API 能做什麼？")
            print(f"A: {answer.split(marker)[-1].split('<|im_end|>')[0].strip()}")
        else:
            print(f"A: (no marker) {answer[-200:]}")

        # Add new documents incrementally
        print("\n--- Adding 3 new documents ---")
        t0 = time.time()
        new_ids = engine.add_documents([
            "Gemini API 支援圖片生成、TTS 語音合成、影片生成（Veo 3.1）、音樂生成（Lyria）和語音轉文字",
            "llama.cpp 是一個 C++ 實作的 LLM 推理引擎 支援 GGUF 量化格式",
            "Anthropic 2026 年營收突破 30B USD Claude Code 貢獻 2.5B ARR",
        ])
        dt = time.time() - t0
        print(f"Added docs {new_ids} in {dt:.1f}s")

        # Query after adding
        print("\n--- After add_documents ---")
        texts, _, _ = engine.generate("Gemini API 能做什麼？", require_recall_topk=True)
        answer = texts[0]
        if marker in answer:
            print(f"Q: Gemini API 能做什麼？")
            print(f"A: {answer.split(marker)[-1].split('<|im_end|>')[0].strip()}")
        else:
            print(f"A: (no marker) {answer[-200:]}")

        texts, _, _ = engine.generate("Anthropic 營收多少？", require_recall_topk=True)
        answer = texts[0]
        if marker in answer:
            print(f"\nQ: Anthropic 營收多少？")
            print(f"A: {answer.split(marker)[-1].split('<|im_end|>')[0].strip()}")
        else:
            print(f"\nA: (no marker) {answer[-200:]}")

    print("\nDone!")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
