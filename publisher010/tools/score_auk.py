"""English SeedTTS mean utterance WER using the original Whisper-large-v3 protocol.

All generated utterances are required; no filtering of high-WER samples.
Whisper language/task kwargs replace the obsolete forced_decoder_ids API only.
"""
from pathlib import Path
import hashlib
import json
import string
import time

BASE=Path('/root/private_data/guoxin-session010')
OUT=BASE/'auk-scoring001'


def edit_distance(reference,hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def normalized(text,punctuation):
    for character in punctuation:
        if character!="'":text=text.replace(character,'')
    return text.replace('  ',' ').lower().split()


def main():
    import numpy as np
    import torch
    import soundfile as sf
    from scipy.signal import resample
    from zhon.hanzi import punctuation
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    assert (BASE/'auk-generation001/GENERATION-COMPLETE.json').is_file()
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=False)
    model_path=str(BASE/'whisper-large-v3')
    processor=WhisperProcessor.from_pretrained(model_path,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(model_path,local_files_only=True,dtype=torch.float32).to('cuda').eval()
    samples=[json.loads(x) for x in (BASE/'seedtts/samples.jsonl').read_text().splitlines()]
    assert len(samples)==1088
    summaries=[];started=time.perf_counter()
    for repetition in (1,2,3):
        input_dir=BASE/f'auk-generation001/run{repetition}'
        generated={r['id']:r for r in (json.loads(x) for x in (input_dir/'requests.jsonl').read_text().splitlines())}
        assert len(generated)==1088 and set(generated)=={s['id'] for s in samples}
        values=[]
        with (OUT/f'run{repetition}.jsonl').open('x') as log:
            for sample in samples:
                record=generated[sample['id']];assert record['status']=='ok'
                file=BASE/'auk-generation001'/record['file']
                assert hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256']
                audio,sr=sf.read(file)
                if sr!=16000:audio=resample(audio,int(len(audio)*16000/sr))
                features=processor(audio,sampling_rate=16000,return_tensors='pt').input_features.to('cuda')
                with torch.inference_mode():
                    ids=model.generate(features,language='english',task='transcribe',do_sample=False)
                transcript=processor.batch_decode(ids,skip_special_tokens=True)[0]
                truth=normalized(sample['text'],punctuation+string.punctuation)
                hypothesis=normalized(transcript,punctuation+string.punctuation)
                assert truth
                distance=edit_distance(truth,hypothesis)
                value={'id':sample['id'],'reference':sample['text'],'transcript':transcript,'normalized_reference':truth,
                       'normalized_transcript':hypothesis,'errors':distance,'reference_words':len(truth),
                       'wer':distance/len(truth),'audio_sha256':record['sha256']}
                values.append(value);log.write(json.dumps(value)+'\n');log.flush()
                if len(values)%32==0:print(json.dumps({'scoring_run':repetition,'done':len(values)}),flush=True)
        summary={'run':repetition,'samples':len(values),'mean_utterance_wer_percent':100*sum(x['wer'] for x in values)/len(values),
                 'corpus_wer_percent':100*sum(x['errors'] for x in values)/sum(x['reference_words'] for x in values),
                 'over_50_percent_wer':sum(x['wer']>.5 for x in values),'excluded_samples':0}
        summaries.append(summary)
    mean=sum(s['mean_utterance_wer_percent'] for s in summaries)/3
    result={'complete':True,'runs':summaries,'three_run_mean_wer_percent':mean,'published_test_en_wer_percent':1.02,
            'threshold_wer_percent':1.0404,'numeric_comparison_pass':mean<=1.0404,
            'wall_seconds':time.perf_counter()-started,'speaker_similarity_evaluated':False,
            'official_acceptance':False,'submitted':False,'whisper_revision':'06f233fe06e710322aca913c1bc4249a0d71fce1',
            'protocol_note':'Full1088-row English replica recommended by SGLang-Omni; source paper describes approximately1000; original unfiltered per-utterance mean WER normalization.'}
    (OUT/'verify_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
