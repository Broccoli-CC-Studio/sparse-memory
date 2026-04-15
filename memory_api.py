"""
MSA Memory API — CRUD wrapper around MSAEngine

Usage:
    from memory_api import MemoryStore

    store = MemoryStore(model_path="ckpt/MSA-4B")
    store.add("菜花住在 Docker 容器裡")
    store.add("RTX 3090 有 24GB VRAM")
    answer = store.query("菜花住在哪裡？")
    store.remove(0)
    store.close()
"""

import json
import os
import sys
import time
import pathlib
import multiprocessing as mp

project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.msa_service import MSAEngine
from src.config.memory_config import GenerateConfig, ModelConfig, MemoryConfig


class MemoryStore:
    def __init__(
        self,
        model_path: str = None,
        kv_cache_dir: str = None,
        doc_top_k: int = 10,
        max_generate_tokens: int = 512,
    ):
        if model_path is None:
            model_path = str(project_root / "ckpt" / "MSA-4B")

        if kv_cache_dir:
            os.environ["MSA_KV_CACHE_DIR"] = kv_cache_dir

        # read model config
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        msa_cfg = cfg.get("msa_config", {})

        self.model_path = model_path
        self.doc_top_k = doc_top_k
        self.documents = []  # list of strings
        self.deleted_ids = set()
        self._engine = None
        self._max_generate_tokens = max_generate_tokens
        self._pooling_kernel_size = msa_cfg.get("pooling_kernel_size", 64)
        self._router_layer_idx = msa_cfg.get("router_layer_idx", "all")

    def add(self, text: str) -> int:
        """Add a document to memory. Returns doc_id."""
        doc_id = len(self.documents)
        self.documents.append(text)
        self._dirty = True
        return doc_id

    def add_batch(self, texts: list) -> list:
        """Add multiple documents. Returns list of doc_ids."""
        ids = []
        for t in texts:
            ids.append(self.add(t))
        return ids

    def remove(self, doc_id: int):
        """Lazy delete a document by ID."""
        if doc_id < 0 or doc_id >= len(self.documents):
            raise IndexError(f"doc_id {doc_id} out of range")
        self.deleted_ids.add(doc_id)
        self._dirty = True

    def update(self, doc_id: int, new_text: str) -> int:
        """Update a document. Returns new doc_id."""
        self.remove(doc_id)
        return self.add(new_text)

    def get(self, doc_id: int) -> str:
        """Get document text by ID."""
        if doc_id in self.deleted_ids:
            raise KeyError(f"doc_id {doc_id} has been deleted")
        return self.documents[doc_id]

    def list_docs(self) -> list:
        """List all active document IDs and texts."""
        return [
            (i, self.documents[i])
            for i in range(len(self.documents))
            if i not in self.deleted_ids
        ]

    def _active_docs(self) -> list:
        """Get active documents as list of strings."""
        return [
            self.documents[i]
            for i in range(len(self.documents))
            if i not in self.deleted_ids
        ]

    def _build_engine(self):
        """Rebuild MSAEngine with current active documents."""
        if self._engine is not None:
            self._engine.__exit__(None, None, None)
            self._engine = None

        active = self._active_docs()
        if not active:
            return

        # write temp memory file
        mem_file = str(project_root / ".memory_bank_tmp.json")
        with open(mem_file, "w") as f:
            json.dump(active, f, ensure_ascii=False)

        effective_top_k = min(self.doc_top_k, len(active))

        generate_config = GenerateConfig(
            devices=[0],
            template="QWEN3_INSTRUCT_TEMPLATE",
            max_generate_tokens=self._max_generate_tokens,
            top_p=0.9,
            temperature=0.0,
            qa_mode=True,
        )
        model_config = ModelConfig(
            model_path=self.model_path,
            doc_top_k=effective_top_k,
            pooling_kernel_size=self._pooling_kernel_size,
            router_layer_idx=self._router_layer_idx,
        )
        memory_config = MemoryConfig(
            block_size=16000,
            pooling_kernel_size=self._pooling_kernel_size,
            slice_chunk_size=16 * 1024,
            memory_file_path=mem_file,
        )

        self._engine = MSAEngine(generate_config, model_config, memory_config)
        self._engine.__enter__()
        self._dirty = False

    def query(self, question: str) -> str:
        """Query the memory store. Returns answer string."""
        if not self._active_docs():
            return "(empty memory)"

        if self._engine is None or getattr(self, '_dirty', True):
            self._build_engine()

        texts, _, _ = self._engine.generate(question, require_recall_topk=True)
        answer = texts[0]

        # extract clean answer
        marker = "The answer to the question is:"
        if marker in answer:
            return answer.split(marker)[-1].split("<|im_end|>")[0].strip()

        parts = answer.split("<|object_ref_end|>")
        if len(parts) > 1:
            return parts[-1].strip()

        return answer.strip()

    def compact(self):
        """Remove deleted docs permanently and rebuild indices."""
        new_docs = self._active_docs()
        self.documents = new_docs
        self.deleted_ids = set()
        self._dirty = True

    def save(self, path: str):
        """Save memory store to disk."""
        data = {
            "documents": self.documents,
            "deleted_ids": list(self.deleted_ids),
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MemoryStore":
        """Load memory store from disk."""
        with open(path) as f:
            data = json.load(f)
        store = cls(**kwargs)
        store.documents = data["documents"]
        store.deleted_ids = set(data.get("deleted_ids", []))
        store._dirty = True
        return store

    def close(self):
        """Shutdown engine."""
        if self._engine is not None:
            self._engine.__exit__(None, None, None)
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return len(self.documents) - len(self.deleted_ids)

    def __repr__(self):
        return f"MemoryStore({len(self)} active docs, {len(self.deleted_ids)} deleted)"
