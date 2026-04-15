"""Feed .claude/memory files into MSA server as documents"""
import glob
import json
import urllib.request

MSA_URL = "http://localhost:8379"
MEMORY_DIR = "/home/agent/.claude/projects/-home-agent/memory"

files = sorted(glob.glob(f"{MEMORY_DIR}/*.md"))
files = [f for f in files if not f.endswith("MEMORY.md")]

texts = []
for f in files:
    with open(f) as fh:
        content = fh.read()
    # strip frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()
    name = f.rsplit("/", 1)[-1].replace(".md", "")
    # truncate very long files to ~2000 chars
    if len(content) > 2000:
        content = content[:2000] + "..."
    texts.append(f"[{name}] {content}")

# check current docs to avoid duplicates
req = urllib.request.Request(f"{MSA_URL}/list")
with urllib.request.urlopen(req) as resp:
    existing = json.loads(resp.read())
existing_texts = {doc[1][:50] for doc in existing["docs"]}

new_texts = [t for t in texts if t[:50] not in existing_texts]
if not new_texts:
    print("No new memories to add")
    exit()

data = json.dumps({"texts": new_texts}).encode()
req = urllib.request.Request(
    f"{MSA_URL}/add_batch",
    data=data,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())

print(f"Added {len(result['doc_ids'])} memory files as docs")
print(f"Doc IDs: {result['doc_ids']}")
