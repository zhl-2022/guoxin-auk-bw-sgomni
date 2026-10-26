"""Add batched positions through the unchanged pinned 1-D rotary operation.

The historical x-transformers 1.31.14 forward accepts only 1-D positions.
No weights, position values, frequency calculation, or sampling settings change.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from types import SimpleNamespace

BASE=Path('/root/private_data/guoxin-session010')


def main():
    destination=BASE/'auk-candidate003'
    assert not destination.exists()
    shutil.copytree(BASE/'auk-candidate',destination,ignore=shutil.ignore_patterns('runtime','__pycache__'))
    path=destination/'vendor/sglang_omni/models/auk/dit.py'
    source=path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='3a735cca04f94b9a73ab31801ce266f44fece8428e27bdaac7d8ad3ce34ec020'
    old='''        freqs, scale = (
            self.rotary_embed.forward_from_seq_len(seq_len)
            if positions is None
            else self.rotary_embed(positions)
        )'''
    new='''        if positions is not None and positions.ndim == 2:
            # x-transformers 1.31.14 only accepts one position row. Preserve
            # its exact per-row arithmetic and stack the independent tables.
            tables = [self.rotary_embed(row) for row in positions.unbind(0)]
            if not all(isinstance(s, (int, float)) and s == 1.0 for _, s in tables):
                raise ValueError("Batched AuK rotary compatibility requires non-XPos scaling")
            freqs = torch.stack([f for f, _ in tables], dim=0)
            scale = 1.0
        else:
            freqs, scale = (
                self.rotary_embed.forward_from_seq_len(seq_len)
                if positions is None
                else self.rotary_embed(positions)
            )'''
    assert source.count(old)==1
    path.write_text(source.replace(old,new))
    sys.path.insert(0,str(destination/'vendor'))
    import torch
    from x_transformers.x_transformers import RotaryEmbedding,apply_rotary_pos_emb
    from sglang_omni.models.auk.dit import AuKDit,Attention
    records=[]
    for device in ('cpu','cuda'):
        if device=='cuda':assert torch.cuda.is_available()
        for batch in (1,4):
            rotary=RotaryEmbedding(64).to(device)
            position=torch.arange(37,device=device).repeat(batch,1)+torch.arange(batch,device=device)[:,None]*7
            dummy=SimpleNamespace(rotary_embed=rotary,qk_fusion=None)
            rope=AuKDit.build_rope(dummy,37,position)
            expected=torch.stack([rotary(row)[0] for row in position])
            assert torch.equal(rope.freqs,expected)
            torch.manual_seed(901)
            q=torch.randn(batch,24,37,64,device=device,dtype=torch.bfloat16)
            k=torch.randn_like(q)
            actual_q,actual_k=Attention.apply_rope(q,k,rope)
            expected_q=torch.cat([apply_rotary_pos_emb(q[i:i+1],expected[i]) for i in range(batch)],0)
            expected_k=torch.cat([apply_rotary_pos_emb(k[i:i+1],expected[i]) for i in range(batch)],0)
            assert torch.equal(actual_q,expected_q) and torch.equal(actual_k,expected_k)
            one=AuKDit.build_rope(dummy,37,None)
            assert torch.equal(one.freqs,rotary.forward_from_seq_len(37)[0])
            records.append({'device':device,'batch':batch,'frequency_exact':True,'bf16_qk_exact':True,'one_dimensional_unchanged':True})
    proof={'passed':True,'cases':records,'before_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
           'after_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_weights_changed':False}
    (BASE/'AUK-BATCH003-READY.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
