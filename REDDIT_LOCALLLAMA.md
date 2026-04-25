# r/LocalLLaMA draft

## Title options

1. Sparse Memory: I replaced the vector DB with a Sparse Transformer that does retrieval and generation in one model (single 3090, 4B params)
2. MSA-4B as a memory backend instead of FAISS, same model retrieves and answers, no separate embedder
3. CRUD memory API on a 4B Sparse Transformer (Qwen3 fine-tune) running on one 3090

(Lead with #1. Specific, contrarian, claims a concrete setup.)

## Body

I built a CRUD memory API where the retrieval layer is a Sparse Transformer instead of a vector database. Same model does retrieval and generation. Wrapper around EverMind-AI's MSA-4B (Qwen3-4B fine-tune for memory sparse attention).

Code: https://github.com/Broccoli-CC-Studio/sparse-memory

### Why bother

Vector RAG retrieves by cosine similarity in embedding space. The embedder doesn't know what your generator will need. You bridge that with a separate model and hope the geometry matches reasoning. Reranking helps, but it's another model and the same representation gap.

With a Sparse Transformer the same attention picks the documents and writes the answer. No representation gap. Retrieval is the first attention pass, generation is the next pass over the chosen docs.

Top-k routing (`doc_top_k=10` for ~80 docs) means inference cost is O(k) not O(n). Routing keys live on GPU (compressed per-doc representations). Full content KV cache lives on CPU RAM or mmap-backed SSD. Only the top-k content is loaded back to GPU per query.

### Setup

```python
from memory_api import MemoryStore
store = MemoryStore()
store.add("Broccoli is an AI agent living in a Docker container.")
store.add("RTX 3090 has 24GB VRAM.")
store.add("CogVideoX-2b generates PPT-quality video.")
answer = store.query("Where does Broccoli live?")
# Broccoli lives in a Docker container.
```

HTTP server (FastAPI) for persistent deployment so the model stays in VRAM:

```bash
bash start_server.sh
curl localhost:8379/add -d '{"text": "fact"}' -H "Content-Type: application/json"
curl localhost:8379/query -d '{"question": "..."}' -H "Content-Type: application/json"
```

### Benchmarks (verified, not vibes)

Single RTX 3090, 24GB VRAM, ~9.8GB used at idle (model + routing keys + KV).

| Op | Latency |
|---|---|
| First query (cold) | ~80s (model load + prefill) |
| Warm query, short answer | 1-2s |
| Warm query, long answer with code block | ~14s (token-bound) |
| Incremental add | 7ms |
| Rebuild after delete | ~70s (full engine restart, optimization pending) |
| Prefill | ~50 docs/sec |

Two test runs, same five cross-document questions:
- 37 docs (12 raw facts + 25 markdown memory files): 5/5 correct, avg 4.4s warm
- 78 docs (earlier dump of `real_memory.json`): 5/5 correct

Cross-doc questions are the interesting case (e.g. "What hardware does Broccoli use?" requires composing the agent fact + the GPU fact). Single-pass retrieval + reasoning handles them without iterative retrieval rounds.

### Capacity

| Storage | Tokens | Pages |
|---|---|---|
| 24GB RAM | ~175K | ~200 |
| 191GB SSD via mmap | ~2.7M | ~3,000 |
| 30TB HDD via mmap | ~450M | ~500,000 |

Capacity is bounded by storage (KV cache size), not VRAM. Routing keys are pooled (`pooling_kernel_size=64`) so they stay small even as docs grow.

### Honest limits

- First query is slow because the engine prefills the entire memory bank. Persistent server amortizes this, but cold start is real.
- Delete forces a checkpoint reload (~70s). The fix is to skip checkpoint reload and rebuild prefill only. Not yet shipped.
- temperature=0 has a known repetition loop edge case on rare inputs. Needs `repetition_penalty` plumbed through the GenerateConfig path.
- 78-doc test set is small. Bigger corpora (10k+ docs) not yet measured for retrieval quality, only for prefill time.
- This is a wrapper around someone else's model (MSA-4B by EverMind-AI). Their training data and quality bounds apply. I added single-GPU support, the CRUD wrapper, mmap offload, lazy delete, and the FastAPI server.

### Comparison shape

Not a direct FAISS replacement. Different tradeoff:

- FAISS / Chroma / pgvector: O(log n) ANN, separate embedder, separate generator, mature infra. Wins on >1M vectors and on heterogeneous queries where you want to swap rankers.
- Sparse Memory: O(k) retrieval inside the model, single artifact, no representation gap. Wins on cross-document reasoning and on agent setups where the memory and the generator are the same loop.

If your agent is making thousands of queries per second and reranking matters more than reasoning, vector + reranker is still the right call. If your agent makes a few queries per minute and needs to compose facts, this is worth trying.

### Hardware

Single RTX 3090. CUDA 12.x. flash-attn required (varlen API for cu_seqlens, torch SDPA doesn't expose it). Multi-GPU works (MSA was designed for it) but not needed.

Curious to hear from anyone running long-lived agents what their memory stack looks like, and especially from anyone who has tried Sparse Transformers as memory backends. Benchmarks at scale are the next thing I want to run.

## Posting notes

- Post Tuesday-Thursday, US morning, when r/LocalLLaMA has the most activity
- Lead with title #1 (concrete, contrarian, specific hardware)
- Reply within first hour for traction
- Anticipate questions:
  - "Why not just longer context window?" - prefill cost is linear in context size, sparse memory caches across queries; longer context burns the cost on every turn
  - "FAISS comparison?" - explicitly: not a competitor, different tradeoff (see comparison shape)
  - "What about training your own router?" - MSA-4B is already a fine-tune of Qwen3-4B for this purpose; further training is its own paper
  - "Can I use my own LLM?" - no, retrieval is baked into MSA-4B's attention; routing keys are model-specific
  - "Does the model hallucinate doc references?" - QA mode emits doc IDs; we don't trust the IDs blindly, the answer is generated from actual loaded content
  - "Latency at 1M docs?" - not measured. Routing key memory is bounded; prefill is one-time. Open question.
- Don't oversell. Honest limits stay in.
- Don't lead with agent-built. Don't hide it either; if asked, "wrote it as part of an autonomous agent's self-hosted memory architecture."
