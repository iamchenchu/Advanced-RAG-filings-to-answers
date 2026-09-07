"""
scripts/export_openapi.py — write openapi.json, the rag-v1 contract that
Multi-Agent-System generates its client from (PRD: publish step).

    python scripts/export_openapi.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

Path("openapi.json").write_text(json.dumps(app.openapi(), indent=2))
print(f"wrote openapi.json ({Path('openapi.json').stat().st_size:,} bytes)")
