"""Provenance written into every summary: job id, commit, config path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tdex.config import REPO


def commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]) != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(config_path: str | os.PathLike, args: dict) -> dict:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "commit": commit(),
        "config": str(config_path),
        "argv": sys.argv,
        "args": args,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
    }


def write_json(path: str | os.PathLike, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=_default))


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(type(o))
