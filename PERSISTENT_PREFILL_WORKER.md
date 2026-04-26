# Persistent PrefillStage1Worker design (P1.1 follow-up)

## Problem

After P1.1 step 5 ships the in-place reset path, every `reset_documents` call still spawns a fresh `PrefillStage1Worker` subprocess that loads the model from disk. Measured cost: ~4.5s per spawn. Verified rebuild baseline of 17.5s breaks down as ~9.5s MSAService reload (skipped by P1.1) + ~4.5s prefill worker spawn + ~2s prefill + ~1s query. P1.1 alone takes wall time from 17.5s to ~8s. Killing the prefill worker spawn would take it to ~3s.

## Approach

Make the prefill worker persistent across resets. Spawn once at MSAEngine init, reuse for each `generate_blocks` call, stop only at engine teardown.

## Current code path

`Memory._start_worker()` at `msa_service.py:355` spawns the subprocess. `_stop_worker()` at `:371` kills it. Both are called by `generate_blocks()` at `:551` and `:618` as a self-contained pair. Worker subprocess runs `PrefillStage1Worker.prefill_worker_main` at `prefill.py:111`, which loads the model, processes one batch of docs, then waits for a CLOSE signal and exits.

## Required changes

### Worker protocol (prefill.py)

Add a new message: `PREFILL_WORKER_BATCH_DONE` to signal end-of-batch without exiting.

Rewrite `prefill_worker_main` to loop on the request queue:

```python
worker = PrefillStage1Worker(...)
send PREFILL_WORKER_READY
while True:
    msg = expect_any(request_queue, [PREFILL_WORKER_MEMORY_DOCS, PREFILL_WORKER_CLOSE])
    if msg.type == PREFILL_WORKER_CLOSE:
        break
    docs = msg.data
    try:
        for block in split_docs(docs, block_size):
            meta = worker.inference(block)
            send PREFILL_WORKER_META
    except Exception as e:
        # log + continue to next batch instead of exiting
        traceback.print_exc()
    send PREFILL_WORKER_BATCH_DONE
print(f"prefill worker {gpu_id} ended")
```

`expect_any` is a small helper that pops a message and returns its tag-and-data. The current `expect(queue, expected_tag)` raises on tag mismatch; the persistent loop needs to dispatch on tag.

### Memory class (msa_service.py)

`_start_worker` runs once during `Memory.__init__`. Add a flag `self._worker_persistent = True`.

`_stop_worker` runs only at `Memory.__exit__` or when explicitly requested.

`generate_blocks` no longer brackets with `_start_worker` / `_stop_worker`. Instead, it sends `PREFILL_WORKER_MEMORY_DOCS` and waits for `PREFILL_WORKER_BATCH_DONE` after the per-block `PREFILL_WORKER_META` stream.

```python
def generate_blocks(self, docs):
    # (skip _start_worker; worker already running)
    self.block_desc.init_docs(docs, self.device)
    PrefillStage1Worker.send_documents(self._worker_req_q, docs)

    pbar = tqdm(...)
    recv = 0
    total_docs = len(docs)
    while recv < total_docs:
        meta = PrefillStage1Worker.recv_meta(self._worker_rsp_q)
        recv += meta['nr_docs']
        pbar.update(meta['nr_docs'])
        # ... existing copy-into-blocks logic ...
    pbar.close()

    # NEW: wait for batch-done signal so the worker is idle and ready
    PrefillStage1Worker.recv_batch_done(self._worker_rsp_q)

    # (skip _stop_worker; worker stays alive)
    self.block_desc.merge_poolig_doc_id(self.device)
    if memory_data_path:
        self.serialize(memory_data_path)
    self._post_process()
```

### Memory.__init__ ordering

Currently `_start_worker` is called inside `generate_blocks`. To make persistent, call `_start_worker` before the first `generate_blocks` call. Two options:

1. Move into `Memory.__init__`. Worker starts immediately on engine init, even before any docs are loaded. Simple.
2. Lazy-start on first `generate_blocks`. Defers cost until needed but adds branching.

Option 1 is cleaner. Cost is ~4.5s extra at engine init, but engine init already takes ~80s on cold start; relative cost is small.

### Reset compatibility

`reset_memory()` does not need to touch the prefill worker. The worker holds no per-doc state (model and tokenizer only). After reset, `generate_blocks` is called with the new doc set; worker handles it like any other batch.

## Cost estimate

Wall time after this fix:

- Cold start: unchanged (~80s, model loads in worker once during init)
- Reset (`reset_documents`): ~3s. Breakdown: 0s prefill worker spawn + ~2s prefill + ~1s query.

Total speedup vs original: 17.5s -> 3s, ~6x.

## Risks

- **Orphan subprocess on parent SIGTERM.** When the persistent worker stays alive across resets, `mp.Process` defaults to `daemon=False`, so the prefill subprocess survives parent death and leaks ~9 GB of GPU memory. Fix: pass `daemon=True` when constructing the prefill worker process. Verified 2026-04-26 19:00: `/shutdown` now leaves GPU at 94 MB (no orphan).
- **Worker memory leak across batches.** PrefillStage1Worker creates `past_key_values` per batch in `_inference`. If the cache is not freed between batches, GPU memory grows. The current code does `torch.cuda.empty_cache()` after each batch (`prefill.py:298`), but that may not free everything. Need to monitor `nvidia-smi` over multiple resets.
- **Worker crash mid-batch.** Current code exits on exception; persistent code logs and continues. A crashed worker that returned a partial batch leaves the engine in an inconsistent state. Need to make the engine detect partial-batch and re-issue.
- **Queue ordering.** With persistent workers, two consecutive `generate_blocks` calls could overlap if the first hasn't finished when the second starts. The engine's existing `_worker_all_gather` handles this for other commands; need similar gating here.

## Implementation status (shipped 2026-04-26 19:00)

All 7 steps below landed plus a `daemon=True` fix. Verified end-to-end against a 35-doc state file: cold 14.77s, warm 4.15s, post-delete 7.77s, `/shutdown` returns GPU to 94 MB with no orphan PIDs.

1. `PREFILL_WORKER_BATCH_DONE` constant in `prefill.py:36`
2. `recv_batch_done` static helper next to `recv_meta` in `prefill.py:111`
3. Persistent dispatch loop in `prefill_worker_main` at `prefill.py:115`
4. `_start_worker` made idempotent in `msa_service.py:355` (lazy-start on first `generate_blocks`)
5. `_stop_worker` call removed from `generate_blocks`; replaced by `recv_batch_done`
6. `recv_batch_done` after per-block recv loop in `generate_blocks` at `msa_service.py:619`
7. `_stop_worker` made idempotent for safe `__exit__` use
8. Prefill `mp.Process` now constructed with `daemon=True` so parent SIGTERM auto-kills the persistent subprocess

Run-to-run answer-length variance dominates the post-delete comparison; the architectural saving (~4.5s subprocess spawn skipped on every reset after the first) is real but easier to see in `server.log` (one "Prefill worker 0 is ready" line covers many resets) than in wall-clock latency.

## What this design does NOT cover

- True streaming prefill (overlap doc tokenization with inference). Not free; needs torch.compile or CUDA streams.
- Multi-batch concurrent reset (call `reset_documents` from two threads). Single-threaded only.
- Hot-swap of model weights (would require unloading + reloading, defeating the purpose).
