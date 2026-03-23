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