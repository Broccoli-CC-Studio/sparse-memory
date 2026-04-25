"""Feed .claude/memory files into MSA server as documents.

Default mode: idempotent add. Skips any source file whose `[name]` prefix
already matches an existing doc. Safe to re-run.

Update mode (`--update`): upserts. For each source file, if a doc with the
same `[name]` prefix exists but the body has changed, delete the stale doc
and add the fresh one. New files get added. Use this after editing memory
files to propagate the change without manually finding and deleting the
old doc.

    uv run python3 feed_memories.py            # default, idempotent add
    uv run python3 feed_memories.py --update   # sync source files into store
"""
import glob
import json
import sys
import urllib.request

MSA_URL = "http://localhost:8379"
MEMORY_DIR = "/home/agent/.claude/projects/-home-agent/memory"

UPDATE = "--update" in sys.argv[1:]


def _http_get(path):
    with urllib.request.urlopen(f"{MSA_URL}{path}") as resp:
        return json.loads(resp.read())


def _http_post(path, payload):
    req = urllib.request.Request(
        f"{MSA_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _http_delete(path):
    req = urllib.request.Request(f"{MSA_URL}{path}", method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# --- 1. Build source set from memory files ---

files = sorted(glob.glob(f"{MEMORY_DIR}/*.md"))
files = [f for f in files if not f.endswith("MEMORY.md")]

source = {}  # name -> body text  (body = "[name] ...")
for f in files:
    with open(f) as fh:
        content = fh.read()
    # strip frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()
    name = f.rsplit("/", 1)[-1].replace(".md", "")
    if len(content) > 2000:
        content = content[:2000] + "..."
    source[name] = f"[{name}] {content}"


# --- 2. Read existing docs from server ---

existing = _http_get("/list")["docs"]  # list of [doc_id, text]
# index existing by name prefix `[name]`
existing_by_name = {}
for doc_id, text in existing:
    if text.startswith("[") and "]" in text:
        name = text[1:].split("]", 1)[0]
        existing_by_name[name] = (doc_id, text)


# --- 3. Compute diff ---

to_add = []          # new texts (no matching name)
to_update = []       # (old_doc_id, new_text) for changed body
unchanged = 0

for name, fresh in source.items():
    if name in existing_by_name:
        old_id, old_text = existing_by_name[name]
        if old_text == fresh:
            unchanged += 1
        else:
            to_update.append((old_id, fresh))
    else:
        to_add.append(fresh)


# --- 4. Apply ---

if not UPDATE:
    if to_update:
        print(f"WARN: {len(to_update)} memory files have changed since last sync")
        print(f"      run with --update to propagate")
    if not to_add:
        print(f"No new memories to add ({unchanged} unchanged, {len(to_update)} stale)")
        sys.exit()
    result = _http_post("/add_batch", {"texts": to_add})
    print(f"Added {len(result['doc_ids'])} new memory files (default mode skips updates)")
    print(f"Doc IDs: {result['doc_ids']}")
    sys.exit()

# update mode
if not (to_add or to_update):
    print(f"In sync ({unchanged} docs match source files)")
    sys.exit()

# delete stale docs first so they do not collide with re-add
for old_id, _ in to_update:
    _http_delete(f"/remove/{old_id}")

new_texts = to_add + [text for _, text in to_update]
if new_texts:
    result = _http_post("/add_batch", {"texts": new_texts})
    new_ids = result["doc_ids"]
else:
    new_ids = []

print(f"Synced: {len(to_add)} added, {len(to_update)} updated, {unchanged} unchanged")
if new_ids:
    print(f"New doc IDs: {new_ids}")
