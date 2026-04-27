"""Diff two FP8 parity benchmark JSON outputs."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

n = len(a["queries"])
identical = 0
near = 0
diff = 0
lat_a = 0
lat_b = 0

for qa, qb in zip(a["queries"], b["queries"]):
    aa, ab = qa["a"], qb["a"]
    if aa == ab:
        identical += 1
    else:
        common = sum(1 for c in aa if c in ab) / max(len(aa), 1)
        if common > 0.8:
            near += 1
            tag = "NEAR"
        else:
            diff += 1
            tag = "DIFF"
        print(f'{tag}: Q={qa["q"]}')
        print(f'  {a["dtype"]}: {aa}')
        print(f'  {b["dtype"]}: {ab}')
    lat_a += qa["latency_s"]
    lat_b += qb["latency_s"]

print()
print(f'Total: {n} queries / {a["dtype"]} vs {b["dtype"]}')
print(f'Byte-identical: {identical}/{n}')
print(f'Near (>80% char overlap): {near}/{n}')
print(f'Different: {diff}/{n}')
print(f'{a["dtype"]} mean latency (warm): {sum(q["latency_s"] for q in a["queries"][1:])/(n-1):.2f}s')
print(f'{b["dtype"]} mean latency (warm): {sum(q["latency_s"] for q in b["queries"][1:])/(n-1):.2f}s')
