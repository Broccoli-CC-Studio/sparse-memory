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

        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        msa_cfg = cfg.get("msa_config", {})

        self.model_path = model_path
        self.doc_top_k = doc_top_k
        self.documents = []
        self.deleted_ids = set()
        self.core_summary = ""  # global context prepended to every query
        self._pending_adds = []  # texts waiting to be incrementally prefilled
        self._engine = None
        self._needs_rebuild = False  # true after delete — requires full rebuild on next query
        self._max_generate_tokens = max_generate_tokens
        self._pooling_kernel_size = msa_cfg.get("pooling_kernel_size", 64)
        self._router_layer_idx = msa_cfg.get("router_layer_idx", "all")

    def add(self, text: str) -> int:
        doc_id = len(self.documents)
        self.documents.append(text)
        self._pending_adds.append(text)
        return doc_id

    def add_batch(self, texts: list) -> list:
        return [self.add(t) for t in texts]

    def remove(self, doc_id: int):
        if doc_id < 0 or doc_id >= len(self.documents):
            raise IndexError(f"doc_id {doc_id} out of range")
        self.deleted_ids.add(doc_id)
        self._needs_rebuild = True

    def update(self, doc_id: int, new_text: str) -> int:
        self.remove(doc_id)
        return self.add(new_text)

    def update_core(self, text: str):
        self.core_summary = text

    def get(self, doc_id: int) -> str:
        if doc_id in self.deleted_ids:
            raise KeyError(f"doc_id {doc_id} has been deleted")
        return self.documents[doc_id]

    def list_docs(self) -> list:
        return [
            (i, self.documents[i])
            for i in range(len(self.documents))
            if i not in self.deleted_ids
        ]

    def _active_docs(self) -> list:
        return [
            self.documents[i]
            for i in range(len(self.documents))
            if i not in self.deleted_ids
        ]

    def find_exact_duplicates(self) -> list:
        """Return groups of active doc_ids whose whitespace-normalized text
        matches, oldest first within each group. Pair with dedupe_exact() to
        drop older copies of dialogue fragments or note files re-ingested
        through different paths. Each returned list has length >= 2.
        """
        groups = {}
        for doc_id, text in self.list_docs():
            normalized = " ".join(text.split())
            groups.setdefault(normalized, []).append(doc_id)
        return [sorted(ids) for ids in groups.values() if len(ids) > 1]

    def dedupe_exact(self) -> int:
        """For every equivalence class of duplicate active docs, keep the
        newest (highest doc_id) and remove the rest. Returns the number of
        docs removed. The next query rebuilds via reset_documents (or
        _build_engine if engine not yet loaded)."""
        removed = 0
        for group in self.find_exact_duplicates():
            for older_id in group[:-1]:
                if older_id not in self.deleted_ids:
                    self.deleted_ids.add(older_id)
                    self._needs_rebuild = True
                    removed += 1
        return removed

    def _build_engine(self, docs):
        """Full rebuild: start engine with given docs."""
        if self._engine is not None:
            self._engine.__exit__(None, None, None)
            self._engine = None

        if not docs:
            return

        mem_file = str(project_root / ".memory_bank_tmp.json")
        with open(mem_file, "w") as f:
            json.dump(docs, f, ensure_ascii=False)

        effective_top_k = min(self.doc_top_k, len(docs))

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

    def _ensure_ready(self):
        """Ensure engine is ready for queries. Use incremental path when possible."""
        active = self._active_docs()
        if not active:
            return

        if self._engine is None:
            # first boot: build with all active docs (loads checkpoint)
            self._build_engine(active)
            self._pending_adds.clear()
            self._needs_rebuild = False

        elif self._needs_rebuild:
            # delete happened: in-place reset, reuse loaded model (no checkpoint reload)
            self._engine.reset_documents(active)
            self._pending_adds.clear()
            self._needs_rebuild = False

        elif self._pending_adds:
            # only new adds: use incremental prefill (no rebuild!)
            self._engine.add_documents(self._pending_adds)
            self._pending_adds.clear()

    def query(self, question: str) -> str:
        if not self._active_docs():
            return "(empty memory)"

        self._ensure_ready()

        if self.core_summary:
            full_question = f"Context: {self.core_summary}\n\nQuestion: {question}"
        else:
            full_question = question

        texts, _, _ = self._engine.generate(full_question, require_recall_topk=True)
        answer = texts[0]

        marker = "The answer to the question is:"
        if marker in answer:
            return answer.split(marker)[-1].split("<|im_end|>")[0].strip()

        parts = answer.split("<|object_ref_end|>")
        if len(parts) > 1:
            return parts[-1].strip()

        return answer.strip()

    def save(self, path: str):
        data = {
            "documents": self.documents,
            "deleted_ids": list(self.deleted_ids),
            "core_summary": self.core_summary,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "MemoryStore":
        with open(path) as f:
            data = json.load(f)
        store = cls(**kwargs)
        store.documents = data["documents"]
        store.deleted_ids = set(data.get("deleted_ids", []))
        store.core_summary = data.get("core_summary", "")
        return store

    def close(self):
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
