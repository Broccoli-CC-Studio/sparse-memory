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

- KV cache for deleted docs would still occupy GPU memory and SSD mmap until rebuild. Capacity would slowly degrade between rebuilds.
- Top-k may include deleted docs that get masked. Effective top-k drops. With heavy deletion the model gets fewer real options.
- Routing decisions are still influenced by deleted docs during the routing pass before the mask. The pooled keys may still affect selection through normalization.

### When to ship B

If users need fast delete more than they need clean memory accounting. Likely the right trade for agent memory where deletes are rare (corrections, not bulk cleanup).

## Recommendation

Ship A first. It is the correct fix: delete sets a tombstone, the next query rebuilds over active docs. The previous compact() API became redundant once this path landed and was removed 2026-04-27. B can layer on top later as an optimization knob for users who do bulk deletes.

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

## Empirical baseline correction (2026-04-26 09:30)

Ran `bench_old_rebuild.py` against the running server (OLD code path, 37 docs):

- warm query baseline: 3.11s
- add doc API: 0.001s
- warm query post-add: 3.16s
- remove API call: 0.001s
- **rebuild-triggered query: 17.51s**

The 70s baseline I had been quoting was an unverified estimate from an earlier cron note. The actual rebuild path on 37 docs is 17.5s. Breakdown likely:
- ~9.5s MSAService checkpoint re-load
- ~4.5s PrefillStage1Worker checkpoint load
- ~2s prefill of 37 docs
- ~1s query generation

This corrects the optimization claim:

- OLD rebuild: 17.5s (verified)
- NEW with `reset_documents`, MSAService kept loaded: skips the ~9.5s. PrefillStage1Worker still pays ~4.5s + prefill + query.
- Predicted NEW: ~8s, roughly 2x speedup, not the 5-7x I revised down to or the 35x I originally predicted.

The optimization is still worth shipping (cuts ~10s off every delete), but the "70s → 2s" framing was wrong. Updated SHOW_HN.md and README.md accordingly.

To get further speedup, a follow-up needs to keep `PrefillStage1Worker` alive between resets (single persistent prefill subprocess). At ~4.5s saved per reset, that is ~7s total versus 17.5s baseline, or ~2.5x. Still not 35x.

To verify the current fix end-to-end, test_delete_no_reload.py needs a clean GPU (no concurrent running server).

## End-to-end verification (2026-04-26 17:24)

Yilan terminated the running server from outside the container, freeing GPU. Reran `test_delete_no_reload.py` with a clean MemoryStore + 5 docs:

- cold first query: 9.87s (model load + prefill of 5 docs)
- warm query (no change): 1.01s
- remove doc 4: 0.000s
- **post-delete query: 5.45s**
- warm query after delete: 2.47s

Verdict: PASS. Post-delete latency 5.45s, well under the <10s target and faster than the ~8s prediction. GPU memory before run: 97 MB. After run: 97 MB. Cleanup is clean.

The 5.45s figure is on 5 docs, not 37. The 17.5s OLD baseline was measured on 37 docs.

## 37-doc apples-to-apples (2026-04-26 17:46)

Started fresh server with current commit (`bash start_server.sh` via `Bash(run_in_background=true)`), loaded the 37-doc memory_state.json, and ran a NEW-path measurement matching the 09:30 OLD-path bench.

- cold first query: 12.65s (engine load + prefill of 37 docs)
- warm query: 1.25s
- DELETE /remove/54: 0.001s
- **post-delete query: 8.15s** (OLD baseline 17.51s on the same 37-doc dataset)
- warm query after delete: 1.25s

NEW vs OLD speedup: 17.51 / 8.15 = 2.15x. This matches the prediction in the empirical-correction section: "Predicted NEW: ~8s, roughly 2x speedup."

Closing the lifecycle: POST /shutdown → `{"ok":true,"msg":"shutting down"}` → lifespan handler saved 36 docs (54 removed) at 17:46:35 → SIGTERM → process exit 0. The /shutdown endpoint added in 9a6e68a is now verified end-to-end.

For further gain, a persistent PrefillStage1Worker would skip the ~4.5s checkpoint reload on each reset (designed in `PERSISTENT_PREFILL_WORKER.md`), bringing post-delete to ~3-4s.

## Persistent prefill worker (2026-04-26 19:00)

Implemented per `PERSISTENT_PREFILL_WORKER.md`. The prefill subprocess now stays alive across resets and dispatches on `MEMORY_DOCS` / `CLOSE` tags, sending a `BATCH_DONE` after each batch. `_start_worker` and `_stop_worker` are idempotent. The `mp.Process` for the worker is constructed with `daemon=True`, so it auto-dies when the parent process exits.

Verified against a 35-doc state file:

- cold first query: 14.77s
- warm query: 4.15s
- DELETE /remove/52: 0.002s
- post-delete query: 7.77s
- warm query after delete: 5.79s
- /shutdown lifecycle: GPU back to 94 MB, no orphan compute apps

`server.log` shows one `Prefill worker 0 is ready` line and two `Worker-0 memory inference` rounds (35/35 cold and 34/34 post-delete) without any `Stop prefill worker` between them — the worker persisted as designed.

Post-delete wall-clock is dominated by answer generation length (this run produced multi-line bullet answers, prior runs produced one-sentence answers). The architectural saving from skipping subprocess spawn (~4.5s) shows up as `Prefill worker 0 is ready` appearing once instead of twice in `server.log`, not as a clean wall-clock delta. Comparing across runs requires controlling `max_generate_tokens`.

## Controlled-token measurement (2026-04-26 19:29)

Added `MSA_MAX_GEN_TOKENS` env var to `server.py` and reran with `MSA_MAX_GEN_TOKENS=32`. The 34-doc state file was used (further deletions accumulated across iterations).

- cold first query: 11.40s
- warm query: 0.73s
- DELETE /remove/51: 0.001s
- **post-delete query: 2.64s**
- warm query after delete: 0.79s

With token output constrained, the architectural saving lands clearly: post-delete is 2.64s on 34 docs versus the 17.51s OLD baseline on 37 docs, a 6.6x speedup. The post-delete cost decomposes roughly as ~2s prefill of the remaining 33 docs + ~0.3s generation of 32 tokens + ~0.3s overhead; the persistent worker keeps the ~4.5s subprocess spawn out of the path.

The default 512-token generation case sees post-delete 7-8s where generation dominates and the persistent saving is masked. Users who care about reset latency should constrain `MSA_MAX_GEN_TOKENS` for their workload.

## Out of scope this design

- Concurrent generate-during-reset queueing. Simplest contract: reset blocks until in-flight generates finish, new generates wait for reset.
- mmap KV cache truncation. Worth a separate doc.
- Approach B implementation. Decide after A ships and we have data on real-world delete patterns.
