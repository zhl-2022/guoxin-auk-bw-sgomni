"""Score only after this worker's owned generation service has fully stopped."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import time
import zipfile

BASE=Path('/root/private_data/guoxin-session010')
PYTHON='/root/private_data/guoxin-auk-port006/venv002/bin/python'
DEADLINE=dt.datetime(2026,10,26,20,0,tzinfo=dt.timezone.utc)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['A','B'],required=True);args=ap.parse_args()
    runs=[2] if args.worker=='A' else [1,3]
    out=BASE/('auk-score-controller003-'+args.worker);out.mkdir(exist_ok=False)
    final=BASE/('auk-experiment003-'+args.worker)/'run-final.json'
    record={'worker':args.worker,'runs':runs,'deadline_utc':DEADLINE.isoformat(),'submitted':False}
    try:
        while not final.exists():
            if dt.datetime.now(dt.timezone.utc)>=DEADLINE:raise TimeoutError('Generation not complete before scoring deadline')
            time.sleep(10)
        generation=json.loads(final.read_text())
        assert generation.get('generation_complete') and generation['service_stopped'],generation
        for port in (30000,30001,30002,30119,30129,30139):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        remaining=int((DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds())
        assert remaining>60
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (out/'wer.log').open('xb') as log:
            child=subprocess.Popen([PYTHON,'-u',str(BASE/'auk-tools/score_auk003.py'),'--runs',*map(str,runs)],
                                   env=env,stdout=log,stderr=subprocess.STDOUT)
            try:record['wer_exit']=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                record['wer_exit']=124
        record['wer_complete']=record['wer_exit']==0 and all((BASE/f'auk-scoring003/run{i}-summary.json').exists() for i in runs)
    except Exception as e:record['error']=str(e)
    finally:
        record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        (out/'run-final.json').write_text(json.dumps(record,indent=2)+'\n')
        with zipfile.ZipFile(f'/root/session010-auk-scoring003-{args.worker}.zip','x',zipfile.ZIP_DEFLATED) as z:
            for directory in (out,BASE/'auk-scoring003'):
                if directory.exists():
                    for p in directory.rglob('*'):
                        if p.is_file() and not p.name.endswith('.partial'):z.write(p,p.relative_to(BASE).as_posix())
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
