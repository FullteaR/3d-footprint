"""One cancellable generation. Invoked only by the supervisor."""
from __future__ import annotations

import io
import json
import logging
import resource
import sys
from pathlib import Path

from fastapi import HTTPException, UploadFile

from .api.routes import build_model
from .config import MAX_JOB_BYTES
from .core import cache, report
from .core.export import export_bodies
from .jobs import FORMATS


def run(directory: Path):
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_JOB_BYTES, MAX_JOB_BYTES))
    try:
        report.stage("starting")
        cache.prune()
        params = json.loads((directory / "params.json").read_text())
        svg = directory / "input.svg"
        model = build_model(
            file=UploadFile(io.BytesIO((directory / "input.gpx").read_bytes())),
            plate_svg=UploadFile(io.BytesIO(svg.read_bytes())) if svg.exists() else None,
            **params,
        )
        # Once mesh construction is done, none of the exports read user inputs.
        for name in ("input.gpx", "input.svg"):
            (directory / name).unlink(missing_ok=True)
        report.stage("exporting")
        total = 0
        for fmt in FORMATS:
            data, _, ext = export_bodies(model.bodies, fmt, model.colors,
                                        credit_full=model.credit_full, credit_ascii=model.credit_ascii)
            total += len(data)
            if total > MAX_JOB_BYTES:
                raise ValueError("生成ファイルが大きすぎます。範囲や解像度を小さくしてください。")
            (directory / f"model.{ext}").write_bytes(data)
            del data
        report.write_json(directory / "outcome.json", {"status": "succeeded"})
    except (ValueError, HTTPException) as exc:
        report.write_json(directory / "outcome.json", {
            "status": "failed", "detail": exc.detail if isinstance(exc, HTTPException) else str(exc),
        })
    except Exception as exc:
        logging.error("generation failed: %s", type(exc).__name__)
        report.write_json(directory / "outcome.json", {
            "status": "failed", "detail": "生成に失敗しました。範囲や解像度を小さくして再試行してください。",
        })


if __name__ == "__main__":
    run(Path(sys.argv[1]))
