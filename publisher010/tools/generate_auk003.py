"""Generate the entire fixed SeedTTS English split for three predeclared seeds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import math
import time
import urllib.request

BASE=Path('/root/private_data/guoxin-session010')
OUT=BASE/'auk-generation003'
SEEDS=(1234,2345,3456)


def main():
    import numpy as np
    import soundfile as sf
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=int,choices=[1,2,3],required=True);args=ap.parse_args()
    assert len(args.runs)==len(set(args.runs))
    OUT.mkdir(exist_ok=True)
    data=BASE/'seedtts'
    samples=[json.loads(x) for x in (data/'samples.jsonl').read_text().splitlines()]
    ready=json.loads((data/'READY.json').read_text())
    assert len(samples)==1088 and hashlib.sha256((data/'samples.jsonl').read_bytes()).hexdigest()==ready['samples_sha256']
    for row in samples:
        assert hashlib.sha256((data/row['audio']).read_bytes()).hexdigest()==row['sha256']
    protocol={'seeds':SEEDS,'rows_per_repetition':1088,'concurrency':4,'full_split':True,
              'dataset':ready,'reference_audio_provided':True,'reference_text_provided':True,
              'duration':'Unmodified upstream reference_seconds * UTF8 target/ref bytes',
              'sampling':'AuK base 32 Euler steps,CFG2,sway-1;FP32 DiT storage/BF16 autocast',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol_path=OUT/'protocol.json'
    try:
        with protocol_path.open('x') as f:f.write(json.dumps(protocol,indent=2)+'\n')
    except FileExistsError:
        assert json.loads(protocol_path.read_text())==json.loads(json.dumps(protocol))
    summaries=[]
    for repetition in args.runs:
        seed=SEEDS[repetition-1]
        folder=OUT/f'run{repetition}';folder.mkdir();(folder/'audio').mkdir()
        def request(row):
            record={'id':row['id'],'seed':seed,'reference_sha256':row['sha256'],'status':'failed'}
            started=time.perf_counter()
            try:
                body={'model':'tencent-auk-bw-port006','input':row['text'],'ref_text':row['ref_text'],
                      'ref_audio':'data:audio/wav;base64,'+base64.b64encode((data/row['audio']).read_bytes()).decode(),
                      'seed':seed,'response_format':'wav'}
                encoded=json.dumps(body).encode();record['request_sha256']=hashlib.sha256(encoded).hexdigest()
                request=urllib.request.Request('http://127.0.0.1:30139/v1/audio/speech',encoded,{'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=300) as response:
                    raw=response.read(32*1024*1024)
                    record['content_type']=response.headers.get('Content-Type')
                audio,sr=sf.read(io.BytesIO(raw))
                assert audio.ndim==1 and np.isfinite(audio).all() and np.any(audio!=0)
                duration=len(audio)/sr
                assert abs(duration-row['estimated_output_seconds'])<.1
                target=folder/'audio'/(row['id']+'.wav');target.write_bytes(raw)
                record.update(status='ok',file=target.relative_to(OUT).as_posix(),bytes=len(raw),
                              sha256=hashlib.sha256(raw).hexdigest(),sample_rate=sr,duration_seconds=duration)
            except Exception as exc:
                record.update(error_type=type(exc).__name__,error=str(exc)[:1000])
            record['wall_seconds']=time.perf_counter()-started
            return record
        started=time.perf_counter();records=[]
        with (folder/'requests.jsonl').open('x') as log,ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(request,s) for s in samples]
            for future in as_completed(futures):
                record=future.result();records.append(record);log.write(json.dumps(record)+'\n');log.flush()
                if len(records)%16==0:print(json.dumps({'run':repetition,'done':len(records),'failed':sum(r['status']!='ok' for r in records)}),flush=True)
        summary={'run':repetition,'seed':seed,'count':len(records),'failed':sum(r['status']!='ok' for r in records),
                 'wall_seconds':time.perf_counter()-started,'audio_seconds':sum(r.get('duration_seconds',0) for r in records)}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
        if summary['failed']:raise RuntimeError('Generation failed; no incomplete accuracy result is accepted')
        (folder/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    if all((OUT/f'run{i}/COMPLETE.json').is_file() for i in (1,2,3)):
        all_runs=[json.loads((OUT/f'run{i}/COMPLETE.json').read_text()) for i in (1,2,3)]
        content=json.dumps({'complete':True,'runs':all_runs},indent=2)+'\n'
        destination=OUT/'GENERATION-COMPLETE.json'
        try:
            with destination.open('x') as f:f.write(content)
        except FileExistsError:
            assert destination.read_text()==content


if __name__=='__main__':main()
