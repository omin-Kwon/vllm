# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differentiable text-only GLM from the exact compressed evaluation weights.

Calibration uses dequantized BF16 weights/activations, as the Qwen/Super
calibration loaders do. It does not emulate inference activation quantization.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.glm5_next import modeling_glm5_next as hf


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles = {}
        self.used = set()

    def get(self, key, device):
        shard = self.index[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(self.root / shard, framework="pt")
        self.used.add(key)
        return self.handles[shard].get_tensor(key).to(device)

    def weight(self, prefix, device):
        packed = prefix + ".weight_packed"
        if packed not in self.index:
            value = self.get(prefix + ".weight", device)
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(
                    f"Unexpected unquantized dtype: {prefix}: {value.dtype}"
                )
            return value
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        value = self.get(packed, device)
        value = unpack_fp4_from_uint8(value, value.shape[0], value.shape[1] * 2)
        return dequantize(
            x_q=value,
            scale=self.get(prefix + ".weight_scale", device).to(value.dtype),
            global_scale=self.get(prefix + ".weight_global_scale", device),
            dtype=torch.bfloat16,
        )


def source_name(name):
    for target, source in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        for suffix in ("fn", "base", "scale"):
            name = name.replace(f"{target}.{suffix}", f"{source}_{suffix}")
    return name.replace("self_attn.forget_gate.", "self_attn.")


@torch.no_grad()
def load(root):
    config = AutoConfig.from_pretrained(root).text_config
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.Glm5NextTextModel(config)
    checkpoint = Checkpoint(root)
    loads = [0] * torch.accelerator.device_count()
    devices = []
    for i, layer in enumerate(model.layers):
        device_id = min(range(len(loads)), key=loads.__getitem__)
        device = f"cuda:{device_id}"
        devices.append(device)
        prefix = f"model.language_model.layers.{i}."
        state = {}
        for name, placeholder in layer.state_dict().items():
            if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                value = torch.empty(
                    placeholder.shape, device=device, dtype=torch.bfloat16
                )
                for e in range(config.num_local_experts):
                    if name.endswith("gate_up_proj"):
                        for j, proj in enumerate(("gate_proj", "up_proj")):
                            weight = checkpoint.weight(
                                prefix + f"mlp.experts.{e}.{proj}", device
                            )
                            value[
                                e, j * weight.shape[0] : (j + 1) * weight.shape[0]
                            ].copy_(weight)
                    else:
                        value[e].copy_(
                            checkpoint.weight(
                                prefix + f"mlp.experts.{e}.down_proj", device
                            )
                        )
                state[name] = value
            elif name == "self_attn.conv1d.weight":
                state[name] = torch.cat(
                    [
                        checkpoint.weight(prefix + f"self_attn.{q}_conv1d", device)
                        for q in ("q", "k", "v")
                    ]
                )
            else:
                key = prefix + source_name(name)
                state[name] = (
                    checkpoint.weight(key.removesuffix(".weight"), device)
                    if key.endswith(".weight")
                    else checkpoint.get(key, device)
                )
            if state[name].shape != placeholder.shape:
                raise ValueError(
                    f"Shape mismatch {prefix}{name}: "
                    f"{state[name].shape} != {placeholder.shape}"
                )
        layer.load_state_dict(state, strict=True, assign=True)
        loads[device_id] += sum(t.numel() * t.element_size() for t in state.values())
        del state
        print(
            f"loaded layer {i} on {device}: {loads[device_id] / 2**30:.2f} GiB",
            flush=True,
        )
    model.embed_tokens.load_state_dict(
        {"weight": checkpoint.weight("model.language_model.embed_tokens", devices[0])},
        assign=True,
    )
    model.norm.load_state_dict(
        {"weight": checkpoint.weight("model.language_model.norm", devices[-1])},
        assign=True,
    )
    head = checkpoint.weight("lm_head", devices[-1])
    active = {
        key
        for key in checkpoint.index
        if key == "lm_head.weight"
        or key.startswith("model.language_model.")
        and not key.startswith("model.language_model.layers.45.")
        and not key.endswith(".input_global_scale")
    }
    missing = active - checkpoint.used
    if missing:
        raise ValueError(f"Unconsumed text weights: {sorted(missing)[:20]}")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("Unmaterialized model parameters")
    model.eval().requires_grad_(False)
    return (
        model,
        head,
        devices,
        {
            "used_tensors": len(checkpoint.used),
            "unconsumed_text_weights": 0,
            "device_bytes": loads,
        },
    )


def forward(model, head, devices, ids, grad=True):
    hidden = model.embed_tokens(ids.to(devices[0])).detach().requires_grad_(grad)
    hidden = hidden.unsqueeze(2).expand(-1, -1, model.config.hc_mult, -1).contiguous()
    previous = None
    for i, layer in enumerate(model.layers):
        device = devices[i]
        hidden = hidden.to(device)
        previous = previous.to(device) if previous is not None else None
        hidden, previous = layer(
            hidden,
            attention_mask=torch.ones(ids.shape, device=device, dtype=torch.bool),
            position_ids=torch.arange(ids.shape[1], device=device)[None],
            prev_topk_indices=previous,
            use_cache=False,
        )
    return F.linear(model.norm(model.hc_head(hidden)), head)
