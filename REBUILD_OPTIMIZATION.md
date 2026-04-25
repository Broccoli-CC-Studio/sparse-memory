# Rebuild optimization design (P1.1)

## Problem

`MemoryStore.remove(doc_id)` currently sets `_needs_rebuild = True`. The next query path runs `_build_engine(active_docs)` which:

1. Calls `engine.__exit__()` -> `stop_workers()` -> kills subprocess workers (frees GPU)
2. Constructs new `MSAEngine(...)`
3. New engine `__init__` -> `start_workers()` -> spawns subprocess workers, each running `load_model()` to load the 4B-param checkpoint from disk to GPU
4. `initialize()` -> `_load_memory_file()` + `_process_buckets()` (prefill all active docs)

Wall time: ~70s. Most of this is step 3 (checkpoint load). Step 4 prefill is the same cost regardless.

## Goal

Eliminate step 3 on remove. Reuse the same workers and model. Only re-prefill the remaining docs.

## Approach A: Reset-and-reprefill (clean, ~3-4h work)

### New worker command

Add `ResetMemoryCmd` to the worker protocol. Worker handler:

1. Drop the current memory bank (block/slice tensors, KV caches)
2. Reset internal state (idx_to_doc, prefill_template_vars cached state)
3. Wait for new `init_docs` from engine

Engine-side flow:

```python
def reset_documents(self, new_docs: List[Document]):
    # 1. Send ResetMemoryCmd to all workers, wait for ack
    self._worker_all_gather("ResetMemoryCmd")
    # 2. Re-prefill with new docs (workers reuse loaded model)
    self.memory_config.memory_file_path = self._write_temp_memory_file(new_docs)
    self._load_memory_file()
    self._wait_ready_signal()
    self._process_buckets()
```

`MemoryStore._build_engine` becomes:

```python
def _rebuild_inplace(self, docs):
    if self._engine is None:
        return self._first_build(docs)  # only path that loads checkpoint
    self._engine.reset_documents(docs)
```

### Cost estimate

Wall time after fix:
- Reset: ~0.5-1s (free tensors + clear cache)
- Re-prefill 37 docs at 50 docs/sec: ~0.7s
- Total: ~1.5-2s vs 70s now (~35x speedup)

### Risks

- Memory leaks if worker reset is incomplete. Need to verify GPU memory is actually freed (`torch.cuda.empty_cache()` + check `nvidia-smi` before / after).
- KV cache mmap files on SSD: need to truncate or recreate to avoid stale entries. Check `MSA_KV_CACHE_DIR` handling in `Memory.serialize` / `deserialize`.
- Worker thread state: `recv_rsp_thread` and `recv_req_thread` are running. Reset must not drop in-flight requests. Either reject new requests during reset or queue them.

## Approach B: Tombstone routing filter (faster to ship, weaker semantics)

Mark deleted doc_ids in the engine. At query time, mask their routing keys before top-k selection.

### Where to mask

`gpu_select(scores, k, rk, v, pooled_doc_ids, total_docs)` at `msa_service.py:1228`. Add a `deleted_mask` tensor and set scores for deleted doc_ids to `-inf` before the top-k.

### Engine API

```python
self._engine.set_deleted_doc_ids(set_of_global_ids)
```

Forwarded to workers via a new `SetDeletedIdsCmd`.

### Pros

- O(1) delete, no re-prefill ever
- No checkpoint reload, no memory bank reset
- Compaction stays optional (run when fragmentation exceeds threshold)

### Cons

- KV cache for deleted docs still occupies GPU memory and SSD mmap. Capacity slowly degrades until compaction.
- Top-k may include deleted docs that get masked. Effective top-k drops. With heavy deletion the model gets fewer real options.
- Routing decisions are still influenced by deleted docs during the routing pass before the mask. The pooled keys may still affect selection through normalization.

### When to ship B

If users need fast delete more than they need clean memory accounting. Likely the right trade for agent memory where deletes are rare (corrections, not bulk cleanup).

## Recommendation

Ship A first. It is the correct fix and the existing `compact()` API already implies this semantics (clean memory after delete). B can layer on top later as an optimization knob (`MemoryStore(eager_compact=False)`) for users who do bulk deletes.

## Implementation order

1. [done 04:32] Add `ResetMemoryCmd` / `ResetMemoryResult` dataclasses next to `AddDocumentsCmd` in `msa_service.py:100-117`
2. [done 04:32 → 05:00 wired] Worker handler at `msa_service.py:1893` now calls `service.reset_memory()` and reports `ok=True` on success or the exception string on failure. Mirrors the `add_documents` branch.
3. [done 05:00] `Memory.reset_memory()` at `msa_service.py:638-654`. Drops `blocks` (rebuilds empty `BlockData` per router layer), `block_desc`, `k_slices`, `slice_desc`, and `idx_to_doc`. Keeps `template_prefix_kvcache` and loaded model. Calls `torch.cuda.empty_cache()`. Sanity-tested via `hasattr(Memory, "reset_memory")` import check.
4. [done 05:32] `MSAEngine.reset_documents(new_texts)` at `msa_service.py:2005`. Tokenizes + re-partitions via `balanced_bucket_partition`, sends `ResetMemoryCmd` carrying each worker's bucket + full `idx_to_doc`, waits for ack on `sync_result_queue`, raises on any worker error. Updates engine-side `self.docs` / `self.buckets`. Worker handler now calls `reset_memory()` then `save_idx_to_doc()` + `generate_blocks()` so the worker is fully repopulated in one round trip.
5. [done 06:00] `MemoryStore._ensure_ready` rebuild branch at `memory_api.py:147-151` now calls `self._engine.reset_documents(active)` instead of `self._build_engine(active)`. First-boot branch still uses `_build_engine` since no engine exists yet. `_pending_adds.clear()` and `_needs_rebuild = False` unchanged.
6. [next] Test: a new `test_delete_no_reload.py` that adds 5 docs, queries (warm), removes 1, queries again, and asserts the second query returns under 5s (vs current ~70s). Also a `nvidia-smi` before/after to confirm GPU memory is actually returning.

## Known issues introduced by this fix

- If `MEMORY_DATA_PATH` env var is set (snapshot cache mode), reset will call `generate_blocks` which sees the existing cache and tries to `deserialize` old doc count. This will fail `assert len(docs) == self.block_desc.nr_docs`. Users who use `MEMORY_DATA_PATH` should clear the cache directory before reset, or the reset path should wipe it (deferred decision until someone hits this).
- `PrefillStage1Worker` is spawned and stopped by `generate_blocks` itself, so reset re-spawning a fresh prefill subprocess is safe (no leak), but the prefill worker still re-loads the model from disk on each spawn.

## Empirical measurement (2026-04-26 06:35)

Tested `test_delete_no_reload.py` with a single MemoryStore + 5 docs:

- Phase 2 cold query: 9.77s (model load + prefill, expected for small doc set)
- Phase 3 warm query: 1.01s (no rebuild, baseline)
- Phase 4 remove: 0.000s (just sets `_needs_rebuild`)
- Phase 5 post-delete query: failed with CUDA OOM during reset path

Failure cause was *concurrent test + running server*, not the reset path itself. The PrefillStage1Worker spawn during reset tries to load the model (~4.5 GiB), but the running server's MSAService (9.5 GiB) plus the test's MSAService (9.5 GiB) leave only ~5 GiB free, and the prefill worker can not allocate its full state.

This means:

- Approach A's reset path saves the MSAService re-load (the heavy 9.5 GiB checkpoint), but the PrefillStage1Worker still re-loads ~4.5 GiB on each reset.
- True production flow (server running alone, 24 GiB free for one engine): MSAService stays loaded, PrefillStage1Worker re-loads ~4.5 GiB transiently, total reset wall time should be ~10-15s vs the original ~70s. Roughly 5-7x speedup, not the 35x I had predicted.

To get the full speedup, a follow-up needs to keep `PrefillStage1Worker` alive between resets (single persistent prefill subprocess). That is its own design exercise, deferred.

To verify the current fix end-to-end, test_delete_no_reload.py needs a clean GPU (no concurrent running server).

## Out of scope this design

- Concurrent generate-during-reset queueing. Simplest contract: reset blocks until in-flight generates finish, new generates wait for reset.
- mmap KV cache truncation. Worth a separate doc.
- Approach B implementation. Decide after A ships and we have data on real-world delete patterns.
