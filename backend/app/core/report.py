"""Small per-job progress/warning records, also usable from parse workers.

Each job subprocess receives its own report directory in its environment.
Warnings use fixed source/reason codes; no coordinates, filenames or upstream
URLs are exposed. Separate files let threads and child processes report safely.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def stage(value: str) -> None:
    if directory := os.environ.get("JOB_REPORT_DIR"):
        write_json(Path(directory) / "progress.json", {"stage": value})


def warn(source: str, reason: str = "fetch_failed") -> None:
    if directory := os.environ.get("JOB_REPORT_DIR"):
        warnings = Path(directory) / "warnings"
        if reason == "no_coverage":
            failed = [source]
            if source in {"landuse", "buildings", "bridges", "roads"}:
                failed.append("plateau_catalog")
            if any((warnings / f"{item}-{error}.json").exists()
                   for item in failed for error in ("fetch_failed", "parse_failed")):
                return
        elif reason in {"fetch_failed", "parse_failed"}:
            affected = [source]
            if source == "plateau_catalog":
                affected += ["landuse", "buildings", "bridges", "roads"]
            for item in affected:
                (warnings / f"{item}-no_coverage.json").unlink(missing_ok=True)
        write_json(warnings / f"{source}-{reason}.json",
                   {"source": source, "reason": reason})
