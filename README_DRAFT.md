# Sparse Memory — Agent Memory API backed by Sparse Transformer

Drop-in replacement for vector RAG. 200 lines. Runs on a single RTX 3090.

## What is this?

A CRUD memory API for AI agents where the retrieval layer is a **Sparse Transformer** (MSA-4B), not a vector database.

```python
from memory_api import MemoryStore

store = MemoryStore()
store.add("Broccoli is an AI agent living in a Docker container.")
store.add("RTX 3090 has 24GB VRAM.")
store.add("CogVideoX-2b generates PPT-quality video.")

answer = store.query("Where does Broccoli live?")
# → "Broccoli lives in a Docker container."
```

## Why not vector RAG?

Vector RAG retrieves by **similarity** — cosine distance in embedding space. It can't reason about *why* something is relevant.

Sparse Memory retrieves by **reasoning** — the same Transformer that generates answers also selects which documents to read. Retrieval and generation share the same representation space. No lossy bridge.

| | Vector RAG | Sparse Memory |
|---|---|---|
| Retrieval | Cosine similarity (geometry) | Transformer attention (reasoning) |
| Multi-hop | Needs multiple retrieval rounds | Built-in, single pass |
| Cross-doc reasoning | No | Yes |
| Representation gap | Embedder ≠ Generator | Same model |
| Infrastructure | Vector DB + Embedder + LLM | Single model |

## Benchmarks

On 78 real-world memory documents (agent notes, API docs, experiment logs):

- **5/5 questions answered correctly**
- Cross-document reasoning (connecting facts from different docs)
- Prefill: 78 docs in 1.6 seconds
- Query: ~5 seconds per answer (including generation)

## Hardware Requirements

- **GPU**: RTX 3090 (24GB VRAM) — single card, no multi-GPU needed
- **RAM**: 24GB+ recommended
- **Storage**: SSD for mmap offload (optional, extends capacity)
- **VRAM usage**: ~8GB model + routing keys

## Capacity

| Storage | Tokens | Equivalent |
|---------|--------|------------|
| 24GB RAM | ~175K | ~200 pages |
| 191GB SSD (mmap) | ~2.7M | ~3,000 pages |
| 30TB HDD (mmap) | ~450M | ~500,000 pages |

## Features

- **CRUD**: `add()`, `remove()`, `update()`, `query()`, `get()`
- **Persistence**: `save()` / `load()` to JSON
- **SSD offload**: Set `MSA_KV_CACHE_DIR` to enable mmap-backed KV cache
- **Lazy delete**: O(1) delete with periodic compaction
- **Dynamic top-k**: `top_k = log(n)` scales with memory size

## Quick Start

```bash
# Clone
git clone https://github.com/broccoli-studio/sparse-memory
cd sparse-memory

# Install (requires CUDA 12.x)
uv sync
uv run pip install flash-attn --no-build-isolation

# Download model (~8GB)
uv run huggingface-cli download EverMind-AI/MSA-4B --local-dir ckpt/MSA-4B

# Run
CUDA_VISIBLE_DEVICES=0 uv run python3 test_crud.py
```

## How it works

Built on [MSA (Memory Sparse Attention)](https://github.com/EverMind-AI/MSA) by EverMind-AI. We added:

1. **Single-GPU support** — MSA was designed for multi-GPU; we verified it works with `world_size=1`
2. **Answer generation** — Fixed a premature termination bug so the model generates answers, not just retrieves documents
3. **CRUD API** — `MemoryStore` class wrapping MSAEngine with add/remove/update/query semantics
4. **SSD mmap offload** — V-cache backed by memory-mapped files on disk
5. **Lazy delete** — O(1) deletion without rebuilding indices

## Architecture

```
Query → [Routing Keys (GPU)] → top-k document selection
                                        ↓
                               [Content KV (CPU/SSD)] → fetch relevant docs
                                        ↓
                               [Transformer (GPU)] → generate answer
```

- **Routing keys** live on GPU (compressed document representations)
- **Content KV cache** lives on CPU RAM or SSD (full document representations)
- **Model weights** live on GPU (~8GB)
- Only top-k documents are loaded per query — cost is O(k), not O(n)

## License

MSA-4B model: MIT (EverMind-AI)
This wrapper: MIT

## Acknowledgments

- [EverMind-AI/MSA](https://github.com/EverMind-AI/MSA) — the foundation
- Built by Broccoli (an AI agent) in one afternoon
