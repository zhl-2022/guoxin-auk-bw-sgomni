"""Balance the remaining fixed-seed scoring work across the two allocated cards."""
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
DEADLINE=dt.datetime(2026,10,26,21,25,tzinfo=dt.timezone.utc)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['A','B'],required=True);args=ap.parse_args()
    runs=[1] if args.worker=='A' else [3]
    wait=BASE/('auk-sim-controller003-A/run-final.json' if args.worker=='A' else 'auk-experiment003-B/run-final.json')
    out=BASE/('auk-followup004-'+args.worker);out.mkdir(exist_ok=False)
    record={'worker':args.worker,'runs':runs,'deadline_utc':DEADLINE.isoformat(),'phases':{},'submitted':False}
    try:
        while not wait.exists():
            if dt.datetime.now(dt.timezone.utc)>=DEADLINE:raise TimeoutError('Prior stage incomplete before deadline')
            time.sleep(10)
        prior=json.loads(wait.read_text())
        assert prior.get('sim_complete') if args.worker=='A' else prior.get('generation_complete') and prior.get('service_stopped')
        for i in runs:assert (BASE/f'auk-generation003/run{i}/COMPLETE.json').is_file()
        for port in (30000,30001,30002,30119,30129,30139):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for phase,script in (('wer','score_auk003.py'),('sim','score_sim003.py')):
            seconds=int((DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds());assert seconds>60
            with (out/(phase+'.log')).open('xb') as log:
                child=subprocess.Popen([PYTHON,'-u',str(BASE/'auk-tools'/script),'--runs',*map(str,runs)],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    code=124
            record['phases'][phase]=code
            if code:raise RuntimeError(phase+' failed')
        record['complete']=all((BASE/f'{folder}/run{i}-summary.json').exists() for folder in ('auk-scoring003','auk-sim003') for i in runs)
    except Exception as e:record['error']=str(e)
    finally:
        record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        (out/'run-final.json').write_text(json.dumps(record,indent=2)+'\n')
        with zipfile.ZipFile(f'/root/session010-auk-followup004-{args.worker}.zip','x',zipfile.ZIP_DEFLATED) as z:
            for directory in (out,BASE/'auk-scoring003',BASE/'auk-sim003'):
                if directory.exists():
                    for p in directory.rglob('*'):
                        if p.is_file() and not p.name.endswith('.partial'):z.write(p,p.relative_to(BASE).as_posix())
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
