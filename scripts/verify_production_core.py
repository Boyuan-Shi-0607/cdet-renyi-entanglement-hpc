#!/usr/bin/env python3
"""Verify the immutable production core against the imported upstream commit."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'provenance/upstream_files.json').read_text())
expected = {name: sha for name, sha in manifest['files'].items()
            if name.startswith(('src/', 'include/'))}
actual_names = {str(p.relative_to(root)) for directory in ('src', 'include')
                for p in (root / directory).rglob('*') if p.is_file()}
if actual_names != set(expected):
    raise SystemExit('Production core file inventory differs from upstream')
for name, sha in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
        raise SystemExit(f'Production core changed: {name}')
print(f'PASS: {len(expected)} core files match upstream {manifest["commit"]}')
