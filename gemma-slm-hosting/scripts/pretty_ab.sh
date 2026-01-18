#!/usr/bin/env bash
set -euo pipefail

folder="${1:-artifacts/poc_run}"

python - "$folder" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
base_path = folder / "base.json"
ft_path = folder / "ft.json"

def read_content(path: Path) -> str:
    if not path.exists():
        return f"Missing: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["choices"][0]["message"]["content"].strip()

print("BASE")
print(read_content(base_path))
print()
print("FT")
print(read_content(ft_path))
PY
