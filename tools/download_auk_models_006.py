#!/usr/bin/env python3
"""Transfer the reviewed AuK + Qwen conditioner checkpoint sets. No GPU or pip."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from download_candidate_models_20261023 import check_relative, download_file, json_new

BASE = Path("/root/private_data/guoxin-candidates/models")
IDENTITIES = {"auk006": ("tencent/AuK", "790742b71a4430120daf2b2099192abae449eb9f", "Tencent-Hunyuan/AuK"),
              "qwen25-omni3b006": ("Qwen/Qwen2.5-Omni-3B", "f75b40e3da2003cdd6e1829b1f420ca70797c34e", "Qwen/Qwen2.5-Omni-3B")}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def no_links(path):
    if any(item.is_symlink() for item in (Path(path), *Path(path).parents)):
        raise ValueError("Destination/ancestor symlink refused")


def validate(lock):
    if lock.get("schema") != "session006-auk-fixed-transfer-v1" or len(lock["files"]) != 20:
        raise ValueError("Exact AuK transfer file set differs")
    names = set()
    for row in lock["files"]:
        check_relative(row["path"]); namespace = row["namespace"]
        if (row["original_model"], row["original_revision"], row["modelscope_model"]) != IDENTITIES.get(namespace):
            raise ValueError("Model namespace/revision differs")
        revision = row["modelscope_revision"]
        if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Malformed pinned revision/hash")
        if row["download_url"] != f"https://modelscope.cn/models/{row['modelscope_model']}/resolve/{revision}/{row['path']}":
            raise ValueError("Source URL differs")
        name = (namespace, row["path"])
        if name in names: raise ValueError("Duplicate destination")
        names.add(name)
    return lock["files"]


def apply(rows, max_seconds):
    if sys.platform != "linux": raise ValueError("Apply requires SCNet Linux")
    no_links(BASE); BASE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(BASE).free < sum(x["size"] for x in rows) + 1024**3:
        raise ValueError("Insufficient model storage plus reserve")
    run_id = uuid.uuid4().hex; lock_path = BASE / "auk006-transfer.lock"
    json_new(lock_path, {"run_id": run_id, "pid": os.getpid()})
    report = {"schema": "auk-checkpoints-receipt006-v1", "run_id": run_id, "GPU_imported": False,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "failed", "files": []}
    try:
        deadline = time.monotonic() + max_seconds
        for row in sorted(rows, key=lambda x: x["size"]):
            target = BASE / row["namespace"] / row["path"]; no_links(target); target.parent.mkdir(parents=True, exist_ok=True)
            result = download_file(row, target, row["download_url"], deadline)
            report["files"].append({**result, "path": row["path"], "namespace": row["namespace"], "source": row["download_url"]})
            print(json.dumps({"event": "file_verified", **report["files"][-1]}), flush=True)
        report["status"] = "completed_all_files_verified"
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"; raise
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        json_new(BASE / ("auk006-transfer-" + run_id + ".json"), report)
        if lock_path.is_file() and read(lock_path).get("run_id") == run_id: lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument("--lock", type=Path, required=True); parser.add_argument("--max-seconds", type=int, default=10800)
    args = parser.parse_args(); rows = validate(read(args.lock))
    if args.action == "plan":
        print(json.dumps({"files": len(rows), "total_bytes": sum(x["size"] for x in rows), "GPU_imported": False})); return 0
    if not 60 <= args.max_seconds <= 21600: parser.error("Deadline must be 60..21600 seconds")
    return apply(rows, args.max_seconds)


if __name__ == "__main__": raise SystemExit(main())
