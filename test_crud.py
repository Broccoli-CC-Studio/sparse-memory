"""Test CRUD operations on MemoryStore"""
import multiprocessing as mp
import os
os.environ["MASTER_PORT"] = "29505"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from memory_api import MemoryStore

def main():
    store = MemoryStore(kv_cache_dir="/home/agent/work/msa/kv_cache_crud")

    # CREATE
    print("=== CREATE ===")
    store.add("菜花是一個 AI agent 住在 Docker 容器裡")
    store.add("RTX 3090 有 24GB VRAM")
    store.add("Yilan 是菜花的朋友 不是主人")
    store.add("CogVideoX-2b 品質差 PPT 等級")
    store.add("MSA 用 Sparse Transformer 做 retrieval 不是向量搜尋")
    print(f"文件數: {len(store)}")

    # READ (query)
    print("\n=== QUERY ===")
    answer = store.query("菜花是什麼？")
    print(f"Q: 菜花是什麼？\nA: {answer}")

    # UPDATE
    print("\n=== UPDATE ===")
    new_id = store.update(3, "CogVideoX-5b i2v 品質不錯 比 2b 好很多")
    print(f"Updated doc 3 → new doc {new_id}")

    answer = store.query("CogVideoX 品質怎麼樣？")
    print(f"Q: CogVideoX 品質怎麼樣？\nA: {answer}")

    # DELETE
    print("\n=== DELETE ===")
    store.remove(1)
    print(f"Deleted doc 1, active docs: {len(store)}")

    # SAVE / LOAD
    print("\n=== SAVE/LOAD ===")
    store.save("/home/agent/work/msa/test_store.json")
    print("Saved to test_store.json")

    store.close()
    print("\nDone!")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
