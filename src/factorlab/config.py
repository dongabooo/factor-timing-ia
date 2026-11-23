"""Rutas del proyecto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
REPORTS = PROJECT_ROOT / "reports"

for _p in (DATA_RAW, DATA_INTERIM, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)
