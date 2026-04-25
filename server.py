"""
MSA Memory Service — persistent FastAPI server

Start:
    cd /home/agent/work/msa
    source .venv/bin/activate
    MASTER_PORT=29512 python3 server.py

Endpoints:
    POST   /add          {"text": "..."}           → {"doc_id": 0}
    POST   /add_batch    {"texts": ["...", ...]}    → {"doc_ids": [0, 1, ...]}
    POST   /query        {"question": "..."}        → {"answer": "..."}
    DELETE /remove/{id}                             → {"ok": true}
    POST   /update/{id}  {"text": "..."}           → {"new_doc_id": 5}
    GET    /docs                                    → {"docs": [[0, "..."], ...], "count": 3}
    GET    /doc/{id}                                → {"doc_id": 0, "text": "..."}
    POST   /compact                                 → {"ok": true, "count": 3}
    POST   /save         {"path": "/tmp/x.json"}   → {"ok": true}
    POST   /load         {"path": "/tmp/x.json"}   → {"ok": true, "count": 5}
    GET    /health                                  → {"status": "ok", "docs": 3, "gpu_mb": 9482}
"""

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import sys

os.environ.setdefault("MASTER_PORT", "29512")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from memory_api import MemoryStore


# --- request/response models ---

class AddRequest(BaseModel):
    text: str

class AddBatchRequest(BaseModel):
    texts: list[str]

class QueryRequest(BaseModel):
    question: str

class UpdateRequest(BaseModel):
    text: str

class SaveLoadRequest(BaseModel):
    path: str


# --- app ---

store: MemoryStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    model_path = os.environ.get("MSA_MODEL_PATH", "ckpt/MSA-4B")
    kv_cache_dir = os.environ.get("MSA_KV_CACHE_DIR", "kv_cache")
    doc_top_k = int(os.environ.get("MSA_DOC_TOP_K", "10"))

    store = MemoryStore(
        model_path=model_path,
        kv_cache_dir=kv_cache_dir,
        doc_top_k=doc_top_k,
    )

    # auto-load persisted state if exists
    state_file = os.environ.get("MSA_STATE_FILE", "memory_state.json")
    if os.path.exists(state_file):
        import json
        with open(state_file) as f:
            data = json.load(f)
        store.documents = data["documents"]
        store.deleted_ids = set(data.get("deleted_ids", []))
        n = len(store)
        print(f"Loaded {n} docs from {state_file}")

    global _state_file
    _state_file = state_file

    print(f"MSA Memory Service ready (model={model_path}, top_k={doc_top_k})")
    yield
    # auto-save on shutdown
    if _state_file and len(store) > 0:
        store.save(_state_file)
        print(f"Saved {len(store)} docs to {_state_file}")
    store.close()


app = FastAPI(title="MSA Memory Service", lifespan=lifespan)

_state_file = None

def _auto_save():
    if _state_file:
        store.save(_state_file)


@app.post("/add")
def add_doc(req: AddRequest):
    doc_id = store.add(req.text)
    _auto_save()
    return {"doc_id": doc_id}


@app.post("/add_batch")
def add_batch(req: AddBatchRequest):
    doc_ids = store.add_batch(req.texts)
    _auto_save()
    return {"doc_ids": doc_ids}


@app.post("/query")
def query(req: QueryRequest):
    answer = store.query(req.question)
    return {"answer": answer}


@app.delete("/remove/{doc_id}")
def remove_doc(doc_id: int):
    try:
        store.remove(doc_id)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    _auto_save()
    return {"ok": True}


@app.post("/update/{doc_id}")
def update_doc(doc_id: int, req: UpdateRequest):
    try:
        new_id = store.update(doc_id, req.text)
    except (IndexError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    _auto_save()
    return {"new_doc_id": new_id}


@app.get("/list")
def list_docs():
    docs = store.list_docs()
    return {"docs": docs, "count": len(docs)}


@app.get("/doc/{doc_id}")
def get_doc(doc_id: int):
    try:
        text = store.get(doc_id)
    except (IndexError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"doc_id": doc_id, "text": text}


@app.post("/compact")
def compact():
    store.compact()
    return {"ok": True, "count": len(store)}


@app.post("/save")
def save(req: SaveLoadRequest):
    store.save(req.path)
    return {"ok": True}


@app.post("/load")
def load(req: SaveLoadRequest):
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"file not found: {req.path}")
    import json
    with open(req.path) as f:
        data = json.load(f)
    store.documents = data["documents"]
    store.deleted_ids = set(data.get("deleted_ids", []))
    store._pending_adds.clear()
    store._needs_rebuild = True
    return {"ok": True, "count": len(store)}


@app.get("/health")
def health():
    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
    return {
        "status": "ok",
        "docs": len(store),
        "gpu_mb": gpu_mb,
        "engine_loaded": store._engine is not None,
    }


@app.post("/shutdown")
def shutdown():
    """Save state and trigger graceful shutdown via SIGTERM. Use to free
    the GPU or pick up code changes in src/. The server listens on
    0.0.0.0; deploy behind a firewall if not on a private network."""
    import os
    import signal
    if _state_file and len(store) > 0:
        store.save(_state_file)
    os.kill(os.getpid(), signal.SIGTERM)
    return {"ok": True, "msg": "shutting down"}


if __name__ == "__main__":
    port = int(os.environ.get("MSA_PORT", "8377"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
