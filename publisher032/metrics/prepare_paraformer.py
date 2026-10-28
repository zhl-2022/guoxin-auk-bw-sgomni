"""Fetch the exact CPU metric checkpoint separately from the AuK service."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from download_candidate_models_20261023 import check_relative, download_file, verified_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['plan', 'check', 'download'])
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--max-seconds', type=int, default=3600)
    args = ap.parse_args()
    assert 60 <= args.max_seconds <= 10800
    lock_path = Path(__file__).with_name('paraformer-model-lock.json')
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    assert lock['model'] == 'funasr/paraformer-zh'
    assert lock['revision'] == 'd7811ee3ac581fbcfdeb37c98c6ba674028433dc'
    assert len(lock['files']) == 8
    expected = {'README.md', 'am.mvn', 'config.yaml', 'configuration.json', 'example/asr_example.wav',
                'model.pt', 'seg_dict', 'tokens.json'}
    assert {r['path'] for r in lock['files']} == expected
    rows = [{'path': r['path'], 'size': r['bytes'], 'sha256': r['sha256'],
             'download_url': 'https://huggingface.co/funasr/paraformer-zh/resolve/' + lock['revision'] + '/' + quote(r['path'], safe='/')}
            for r in lock['files']]
    for row in rows:
        check_relative(row['path'])
    plan = {'model': lock['model'], 'revision': lock['revision'], 'files': rows,
            'destination': str(args.destination.absolute()), 'GPU_imported': False, 'AuK_requests': 0}
    if args.action == 'plan':
        print(json.dumps(dict(plan, downloads_executed=False, plan_only=True), indent=2))
        return 0
    destination = args.destination.absolute()
    assert not any(p.is_symlink() for p in (destination, *destination.parents))
    if args.action == 'download':
        destination.mkdir(parents=True, exist_ok=True)
    else:
        assert destination.is_dir()
    deadline = time.monotonic() + args.max_seconds
    receipts = []
    for row in rows:
        target = destination / row['path']
        assert target.resolve().is_relative_to(destination.resolve())
        assert not any(p.is_symlink() for p in (target, *target.parents))
        if args.action == 'download':
            target.parent.mkdir(parents=True, exist_ok=True)
            receipts.append(download_file(row, target, row['download_url'], deadline))
        else:
            assert verified_file(target, row, deadline), row['path']
    ready = destination / 'READY.json'
    if args.action == 'download' and not ready.exists():
        with ready.open('xb') as stream:
            stream.write(lock_bytes)
    if ready.exists():
        actual = json.loads(ready.read_text())
        assert actual['revision'] == lock['revision'] and actual['files'] == lock['files'] and actual['all_files_hashed']
    print(json.dumps({'all8_files_verified': True, 'action': args.action, 'receipts': receipts,
                      'checkpoint_lock_sha256': hashlib.sha256(lock_bytes).hexdigest(),
                      'GPU_imported': False, 'AuK_requests': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
