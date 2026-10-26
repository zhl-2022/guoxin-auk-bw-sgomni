"""Run SIM after this card's WER worker exits, before the platform time cap."""
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
DEADLINE=dt.datetime(2026,10,26,20,5,tzinfo=dt.timezone.utc)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['A','B'],required=True);args=ap.parse_args()
    runs=[2] if args.worker=='A' else [1,3]
    out=BASE/('auk-sim-controller003-'+args.worker);out.mkdir(exist_ok=False)
    record={'worker':args.worker,'runs':runs,'deadline':DEADLINE.isoformat()}
    try:
        wer=BASE/('auk-score-controller003-'+args.worker)/'run-final.json'
        while not wer.exists() or not (BASE/'wavlm-sim/IMPORT-READY.json').exists():
            if dt.datetime.now(dt.timezone.utc)>=DEADLINE:raise TimeoutError('Dependencies incomplete before deadline')
            time.sleep(10)
        assert json.loads(wer.read_text()).get('wer_complete'),json.loads(wer.read_text())
        for port in (30000,30001,30002,30119,30129,30139):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        seconds=int((DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds());assert seconds>60
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (out/'sim.log').open('xb') as log:
            child=subprocess.Popen([PYTHON,'-u',str(BASE/'auk-tools/score_sim003.py'),'--runs',*map(str,runs)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:record['exit']=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                record['exit']=124
        record['sim_complete']=record['exit']==0 and all((BASE/f'auk-sim003/run{i}-summary.json').exists() for i in runs)
    except Exception as e:record['error']=str(e)
    finally:
        record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),submitted=False)
        (out/'run-final.json').write_text(json.dumps(record,indent=2)+'\n')
        with zipfile.ZipFile(f'/root/session010-auk-sim003-{args.worker}.zip','x',zipfile.ZIP_DEFLATED) as z:
            for directory in (out,BASE/'auk-sim003'):
                if directory.exists():
                    for p in directory.rglob('*'):
                        if p.is_file() and not p.name.endswith('.partial'):z.write(p,p.relative_to(BASE).as_posix())
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
