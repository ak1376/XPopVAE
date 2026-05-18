#!/usr/bin/env python3
"""Print one experiment ID per line, sorted — matches Snakefile's EXP_IDS.

When grid.enabled is false (or no dims are defined), the Snakefile produces a
single experiment called "default", and this script prints exactly that.
"""
import sys
import itertools
import re
import yaml
from pathlib import Path


def sanitize(x):
    s = str(x).replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]+", "", s)


vae_yaml = Path(sys.argv[1])
cfg = yaml.safe_load(vae_yaml.read_text())
grid = cfg.get("grid", {})

if not grid.get("enabled", False) or not grid.get("dims"):
    # Matches Snakefile: returns {"default": ...} when grid is off
    print("default")
    sys.exit()

dims = grid["dims"]
prefix = grid.get("name", {}).get("prefix", "experiment")
sep = grid.get("name", {}).get("sep", "__")

ids = []
for combo in itertools.product(*[d["values"] for d in dims]):
    parts = [prefix] + [
        f"{d.get('tag', d['path'].replace('.', '_'))}{sanitize(v)}"
        for d, v in zip(dims, combo)
    ]
    ids.append(sep.join(parts))

for eid in sorted(ids):
    print(eid)
