"""Stop only two owned, still-waiting B queues; preserve their interruption record."""
from pathlib import Path
import datetime as dt
import json
import psutil
import socket
import subprocess
import sys

BASE=Path('/root/private_data/guoxin-session010')


def main():
    # This must run on B while generation is still active and before WER starts.
    assert not (BASE/'auk-score-controller003-B/wer.log').exists()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',30139))==0
    targets=[]
    for script,directory in (('queue_auk_score003.py','auk-score-controller003-B'),('queue_sim003.py','auk-sim-controller003-B')):
        full=str(BASE/'auk-tools'/script)
        matches=[p for p in psutil.process_iter(['cmdline']) if full in (p.info['cmdline'] or []) and p.info['cmdline'][-2:]==['--worker','B']]
        assert len(matches)==1,(script,len(matches))
        process=matches[0];assert not process.children(recursive=True),'Do not interrupt an active scoring process'
        targets.append((process,directory))
    for process,directory in targets:
        process.terminate();process.wait(timeout=15)
        record={'interrupted_by':'rebalance_score_queues004.py','pid':process.pid,'UTC':dt.datetime.now(dt.timezone.utc).isoformat(),
                'reason':'Move run1 scoring to A; B scores run3 only','no_active_scoring_child':True}
        (BASE/directory/'external-interruption004.json').write_text(json.dumps(record,indent=2)+'\n')
    log=(BASE/'auk-followup004-B-controller.log').open('xb')
    child=subprocess.Popen([sys.executable,'-u',str(BASE/'auk-tools/queue_auk_followup004.py'),'--worker','B'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'worker':'B','new_pid':child.pid,'runs':[3]}),flush=True)


if __name__=='__main__':main()
