"""Eight-input reference-audio integration check before full generation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import io
import json
import urllib.request
import urllib.error

BASE=Path('/root/private_data/guoxin-session010')


def main():
    import soundfile as sf
    import numpy as np
    samples=[json.loads(x) for x in (BASE/'seedtts/samples.jsonl').read_text().splitlines()][:8]
    out=BASE/'auk-pilot003';out.mkdir(exist_ok=False)
    def run(sample):
        body={'model':'tencent-auk-bw-port006','input':sample['text'],'ref_text':sample['ref_text'],
              'ref_audio':'data:audio/wav;base64,'+base64.b64encode((BASE/'seedtts'/sample['audio']).read_bytes()).decode(),
              'seed':1234,'response_format':'wav'}
        result={'id':sample['id'],'passed':False}
        try:
            request=urllib.request.Request('http://127.0.0.1:30139/v1/audio/speech',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=300) as r:raw=r.read()
            (out/(sample['id']+'.wav')).write_bytes(raw)
            wave,sr=sf.read(io.BytesIO(raw));duration=len(wave)/sr
            assert wave.ndim==1 and np.isfinite(wave).all() and np.any(wave!=0)
            assert abs(duration-sample['estimated_output_seconds'])<.1
            result.update(passed=True,duration_seconds=duration,sample_rate=sr)
        except urllib.error.HTTPError as e:result.update(error=str(e),body=e.read(4096).decode(errors='replace'))
        except Exception as e:result['error']=repr(e)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,samples))
    (out/'result.json').write_text(json.dumps({'all_passed':all(r['passed'] for r in results),'results':results},indent=2)+'\n')
    print(json.dumps(results),flush=True)
    assert all(r['passed'] for r in results),'Do not continue full generation after integration failure'


if __name__=='__main__':main()
