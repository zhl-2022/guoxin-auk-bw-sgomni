"""Safely preserve the actual session006 result archive and verify every raw member.

Copyright scope notice: no new repository-wide open-source license selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_auk_final007 import checked_zip
from verify_auk_final007_bundle import files_in

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "0b7172198eb446bf875457c3441f089e7715f800e19153aaaf21b002c09e98ba"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/000-session006-auk-results001.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/session006/auk-actual-originals007")
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence/session006/auk-actual-originals007-extraction.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.receipt.exists():
        raise ValueError("Refusing to overwrite previously preserved originals")
    for parent in args.output.parents:
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("Extraction ancestors must not be links")
    members, identity = checked_zip(args.archive, EXPECTED_SHA)
    manifest = json.loads(members["MANIFEST.json"])
    if set(members) != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("Original manifest membership differs from ZIP")
    for name, row in manifest.items():
        raw = members[name]
        if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Original manifest identity differs: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in members.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    if files_in(args.output) != set(members):
        raise ValueError("Extraction membership differs")
    for name, raw in members.items():
        if (args.output / name).read_bytes() != raw:
            raise ValueError("Extraction bytes differ: " + name)
    receipt = {"passed": True, "archive": identity, "CRC_passed": True,
               "manifest_members_verified": len(manifest), "extracted_members_verified": len(members),
               "output": str(args.output), "archive_unmodified": True,
               "scope": "Original ZIP and member integrity; no model execution or scoring"}
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
