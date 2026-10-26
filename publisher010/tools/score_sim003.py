"""Full SeedTTS SIM using the pinned SGLang-Omni WavLM scorer."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

BASE=Path('/root/private_data/guoxin-session010')
sys.path.insert(0,str(BASE/'wavlm-sim'))
from score_auk003 import publish_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=int,choices=[1,2,3],required=True);args=ap.parse_args()
    assert len(args.runs)==len(set(args.runs))
    assets=BASE/'wavlm-sim';assert json.loads((assets/'IMPORT-READY.json').read_text())['ready']
    from speaker_similarity import WavLMSpeakerSimilarity
    import torch
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    scorer=WavLMSpeakerSimilarity(finetune_checkpoint=assets/'wavlm_large_finetune.pth',wavlm_base=assets/'wavlm_large.pt',device='cuda')
    samples=[json.loads(x) for x in (BASE/'seedtts/samples.jsonl').read_text().splitlines()]
    assert len(samples)==len({s['id'] for s in samples})==1088
    out=BASE/'auk-sim003';out.mkdir(exist_ok=True)
    publish_json(out/('protocol-'+ '-'.join(map(str,args.runs))+'.json'),{
        'runs':args.runs,'batch_size':4,'full_split':True,'unit':'cosine_similarity_0_to_1',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256((assets/'speaker_similarity.py').read_bytes()).hexdigest(),
        'source':'sgl-project/sglang-omni@f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88',
        'model_sources':json.loads((assets/'READY.json').read_text()),
        'audio_preprocess':'Pinned SGLang-Omni soundfile float32, scipy resample_poly 16kHz'})
    for run in args.runs:
        folder=BASE/f'auk-generation003/run{run}'
        assert (folder/'COMPLETE.json').exists()
        rows=[json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
        mapping={r['id']:r for r in rows}
        assert len(rows)==len(mapping)==1088 and set(mapping)=={s['id'] for s in samples}
        records=[];started=time.perf_counter()
        with (out/f'run{run}.jsonl').open('x') as f:
            for index in range(0,len(samples),4):
                batch=samples[index:index+4];refs=[];generated=[]
                for s in batch:
                    r=mapping[s['id']];assert r['status']=='ok'
                    ref=BASE/'seedtts'/s['audio'];audio=BASE/'auk-generation003'/r['file']
                    assert hashlib.sha256(ref.read_bytes()).hexdigest()==s['sha256']
                    assert hashlib.sha256(audio.read_bytes()).hexdigest()==r['sha256']
                    refs.append(str(ref));generated.append(str(audio))
                scores=scorer.score_batch(refs,generated)
                assert len(scores)==len(batch)
                for s,value in zip(batch,scores):
                    assert -100.001<=value<=100.001
                    row={'id':s['id'],'similarity':value/100,'audio_sha256':mapping[s['id']]['sha256'],'reference_sha256':s['sha256']}
                    records.append(row);f.write(json.dumps(row)+'\n')
                f.flush()
                if len(records)%64==0:print(json.dumps({'run':run,'sim_done':len(records)}),flush=True)
        publish_json(out/f'run{run}-summary.json',{'run':run,'count':1088,'mean_similarity':sum(r['similarity'] for r in records)/1088,'wall_seconds':time.perf_counter()-started})
    if all((out/f'run{i}-summary.json').exists() for i in (1,2,3)):
        summaries=[json.loads((out/f'run{i}-summary.json').read_text()) for i in (1,2,3)]
        mean=sum(r['mean_similarity'] for r in summaries)/3
        result={'complete':True,'runs':summaries,'three_run_mean_similarity':mean,'publisher_baseline':.788,
                'threshold_relative_2_percent':.788*.98,'numeric_comparison_pass':mean>=.788*.98,
                'official_acceptance':False,'submitted':False}
        publish_json(out/'verify_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
