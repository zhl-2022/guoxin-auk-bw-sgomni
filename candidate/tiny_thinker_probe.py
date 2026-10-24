"""Check the exact AuK Thinker API with a tiny CPU model, never real weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import traceback

def run(processor_path=None):
    import torch
    import transformers
    from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
    class AuKThinker(Qwen2_5OmniThinkerForConditionalGeneration):
        _keys_to_ignore_on_load_unexpected = [
            *(Qwen2_5OmniThinkerForConditionalGeneration._keys_to_ignore_on_load_unexpected or []),
            r"^(talker|token2wav)\.",
        ]
    config = Qwen2_5OmniThinkerConfig(
        text_config={"hidden_size":32,"intermediate_size":64,"num_attention_heads":2,
                     "num_key_value_heads":1,"num_hidden_layers":1,"vocab_size":512,
                     "pad_token_id":0,"bos_token_id":1,"eos_token_id":2,
                     "max_position_embeddings":128,"rope_scaling":{"rope_type":"default","mrope_section":[2,3,3]}},
        audio_config={"d_model":32,"encoder_attention_heads":2,"encoder_ffn_dim":64,
                      "encoder_layers":1,"max_source_positions":64,"num_mel_bins":128,"output_dim":32},
        vision_config={"depth":1,"embed_dim":32,"hidden_size":32,"intermediate_size":64,
                       "num_heads":2,"out_hidden_size":32,"patch_size":14,"spatial_merge_size":2,
                       "temporal_patch_size":2,"fullatt_block_indexes":[0]},
        pad_token_id=0,bos_token_id=1,eos_token_id=2,
        vision_start_token_id=400,vision_end_token_id=401,vision_token_id=407,
        audio_start_token_id=402,audio_end_token_id=403,audio_token_index=404,
        image_token_index=405,video_token_index=406,
    )
    config._attn_implementation = "eager"
    with tempfile.TemporaryDirectory(prefix="auk-tiny-thinker006-") as temp:
        model = AuKThinker(config)
        parameter_count = sum(p.numel() for p in model.parameters())
        model.save_pretrained(temp, safe_serialization=True)
        del model
        # Match the upstream class, from_pretrained keyword and FP32 conversion.
        model = AuKThinker.from_pretrained(temp, torch_dtype=torch.bfloat16,
                                         local_files_only=True, attn_implementation="eager")
        model.visual = None
        model.lm_head = torch.nn.Identity()
        model = model.to(device="cpu", dtype=torch.float32).requires_grad_(False).eval()
        inputs = {"input_ids":torch.tensor([[1, 3, 4]]), "attention_mask":torch.ones(1, 3, dtype=torch.long)}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = torch.stack(outputs.hidden_states, dim=1)
        assert hidden.shape == (1, 2, 3, 32), hidden.shape
        assert torch.isfinite(hidden).all()
        result = {"checked_utc":datetime.now(timezone.utc).isoformat(), "torch":torch.__version__,
                  "transformers":transformers.__version__, "tiny_parameters":parameter_count,
                  "hidden_shape":list(hidden.shape), "finite":True, "checkpoint_weights_loaded":False,
                  "tiny_random_local_save_reload":True,"GPU_execution":False,"passed":True}
    if processor_path:
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path, local_files_only=True)
        messages = [[{"role":"user","content":[{"type":"text","text":"Say hello.<|no_prompt_audio|>"}]}]]
        formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=formatted, padding=True, return_tensors="pt")
        assert inputs["input_ids"].ndim == 2 and inputs["attention_mask"].shape == inputs["input_ids"].shape
        result["real_processor_local_files_only"] = {"path":str(Path(processor_path).resolve()),
             "input_shape":list(inputs["input_ids"].shape), "all_input_keys":list(inputs), "processor_weights_loaded":False}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--processor-path");ap.add_argument("--output",type=Path);args=ap.parse_args()
    try: result=run(args.processor_path)
    except Exception as exc: result={"passed":False,"error":repr(exc),"traceback":traceback.format_exc(),"GPU_execution":False,"checkpoint_weights_loaded":False}
    payload=json.dumps(result,indent=2)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as f:f.write(payload)
    print(payload);return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
