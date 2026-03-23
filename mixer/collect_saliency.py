# mixer/v2_collect_saliency.py
from pathlib import Path
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

TARGETS = {
    "model.layers.{i}.self_attn.q_proj": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj": "blk.{i}.attn_output.weight",
    "model.layers.{i}.mlp.gate_proj": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj": "blk.{i}.ffn_down.weight",
}
N_LAYERS = 36 # arbitrary for now, depends on model/config

def build_target_map():
    out = {}
    for i in range(N_LAYERS):
        for hf_pat, gguf_pat in TARGETS.items():
            out[hf_pat.format(i=i)] = gguf_pat.format(i=i)
    out["lm_head"] = "output.weight"
    return out

def load_chunks(eval_path: str, tokenizer, max_chunks=64, chunk_tokens=256):
    text = Path(eval_path).read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    chunks = []
    step = chunk_tokens
    total = min(len(ids) // step, max_chunks)
    for i in range(total):
        chunk = ids[i*step:(i+1)*step]
        if len(chunk) == step:
            chunks.append(chunk.unsqueeze(0))
    return chunks

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model.to(DEVICE)
    model.eval()

    target_map = build_target_map()
    stats = {
        gguf_name: {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0}
        for gguf_name in target_map.values()
    }

    handles = []

    name_to_module = dict(model.named_modules())

if __name__ == "__main__":
    main()