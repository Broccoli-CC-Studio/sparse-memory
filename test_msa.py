"""
最小化 MSA inference 測試腳本

跑法：
    CUDA_VISIBLE_DEVICES=0 uv run python3 test_msa.py

MSAEngine 會自動把 generate_config.devices 設成 range(torch.cuda.device_count())，
所以只要 CUDA_VISIBLE_DEVICES=0，裡面就只會看到一張卡 (device 0)。
"""

import multiprocessing as mp
import os
import sys
import pathlib

# 把專案根目錄加進 sys.path，讓 src.* import 能找到
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.msa_service import MSAEngine
from src.config.memory_config import GenerateConfig, ModelConfig, MemoryConfig

# ============================================================
# 路徑設定
# ============================================================
MODEL_PATH   = str(project_root / "ckpt" / "MSA-4B")
MEMORY_FILE  = str(project_root / "real_memory.json")

# ============================================================
# 讀取模型自身的 msa_config
# ============================================================
import json
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    _cfg = json.load(f)
_msa = _cfg.get("msa_config", {})

doc_top_k          = 10  # 78 筆真實記憶
pooling_kernel_size = _msa.get("pooling_kernel_size", 64)
router_layer_idx   = _msa.get("router_layer_idx", "all")

# ============================================================
# 設定
# ============================================================
generate_config = GenerateConfig(
    devices=[0],                   # MSAEngine 會 override 成 range(device_count)，這裡只是佔位
    template="QWEN3_INSTRUCT_TEMPLATE",  # 不帶 <think> 的 instruct 模式，輸出更直接
    max_generate_tokens=512,
    top_p=0.9,
    temperature=0.0,
    qa_mode=True,                  # 啟用 QA 模式（帶文件引用格式）
)

model_config = ModelConfig(
    model_path=MODEL_PATH,
    doc_top_k=doc_top_k,
    pooling_kernel_size=pooling_kernel_size,
    router_layer_idx=router_layer_idx,
)

memory_config = MemoryConfig(
    block_size=16000,
    pooling_kernel_size=pooling_kernel_size,
    slice_chunk_size=16 * 1024,
    memory_file_path=MEMORY_FILE,
)

# ============================================================
# 推理
# ============================================================
def main():
    questions = [
        "Gemini API 怎麼生成影片？",
        "fish-speech 支援什麼情緒標籤？",
        "Yilan 是什麼樣的人？",
        "CogVideoX-2b 的品質怎麼樣？",
        "MSA 的容量估算是多少？",
    ]

    with MSAEngine(generate_config, model_config, memory_config) as engine:
        idx_to_doc = engine.get_idx_to_doc()
        print(f"Memory bank 載入完成，共 {len(idx_to_doc)} 筆文件")
        print("=" * 60)

        for q in questions:
            print(f"\n問題: {q}")
            print("-" * 40)
            texts, recall_topks, _ = engine.generate(q, require_recall_topk=True)
            answer = texts[0]
            marker = "The answer to the question is:"
            if marker in answer:
                clean = answer.split(marker)[-1].split("<|im_end|>")[0].strip()
                print(f"回答: {clean}")
            else:
                # try to extract the last part after object_ref_end
                parts = answer.split("<|object_ref_end|>")
                if len(parts) > 1:
                    print(f"回答: {parts[-1].strip()}")
                else:
                    print(f"原始: {answer[:300]}")
            print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
