"""Prepare only the two immutable published metric weights, outside the service."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from download_candidate_models_20261023 import download_file, verified_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'download'])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=7200)
    args = parser.parse_args()
    assert 60 <= args.max_seconds <= 10800
    lock = Path(__file__).with_name('SIM-assets-READY.json').read_bytes()
    files = json.loads(lock)['files']
    assert len(files) == 2
    destination = args.destination.absolute()
    assert not any(p.is_symlink() for p in (destination, *destination.parents))
    rows = []
    for asset in files:
        original = asset['file']
        assert original['path'] in ('wavlm_large_finetune.pth', 'wavlm_large.pt')
        rows.append({'path': original['path'], 'size': original['size'], 'sha256': original['lfs']['oid'],
                     'download_url': 'https://huggingface.co/' + asset['repository'] + '/resolve/' +
                                     asset['revision'] + '/' + quote(original['path'])})
    if args.action != 'plan':
        if args.action == 'download':
            destination.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + args.max_seconds
        for row in rows:
            path = destination / row['path']
            assert not path.is_symlink() and path.resolve().is_relative_to(destination.resolve())
            if args.action == 'download':
                download_file(row, path, row['download_url'], deadline)
            assert verified_file(path, row, deadline), row['path']
    print(json.dumps({'action': args.action, 'files': rows, 'destination': str(destination),
                      'lock_sha256': hashlib.sha256(lock).hexdigest(),
                      'all_weights_verified': args.action != 'plan', 'neural_framework_imported': False,
                      'AuK_generation_requests': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
