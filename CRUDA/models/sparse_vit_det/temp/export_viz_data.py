#!/usr/bin/env python3
"""Runs vit_block_torch.py's encoder once (seed 0) and writes every matrix and
vector of the run as JSON, for the dataflow visualization page.

Each tensor is stored as base64 bytes: int8 scaled by its max |value| (signed
data), or uint8 scaled by its max (probabilities), plus float stats.

Usage: python3 export_viz_data.py out.json [--seed 0]
"""

import argparse
import base64
import json

import numpy as np
import torch

from vit_block_torch import ViTEncoder


def pack(t, kind="div"):
    a = t.detach().double().cpu().numpy()
    if a.ndim == 1:
        a = a[None, :]
    stats = dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))
    if kind == "seq":                                  # non-negative: uint8, 0 .. max
        scale = float(a.max()) or 1.0
        q = np.round(a / scale * 255).astype(np.uint8)
    else:                                              # signed: int8, -maxabs .. maxabs
        scale = float(np.abs(a).max()) or 1.0
        q = np.round(a / scale * 127).astype(np.int8)
    return dict(shape=list(a.shape), kind=kind, scale=scale, stats=stats,
                b64=base64.b64encode(q.tobytes()).decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = ViTEncoder(blocks=12).eval()
    x = torch.randn(144, 384)
    blk = model.blocks[0]

    trace = []
    with torch.no_grad():
        blk(x, trace)
        outs = []
        y = x
        for b in model.blocks:
            y = b(y)
            outs.append(pack(y))
    steps = {name: (t, macs) for name, t, macs in trace}

    def step(name, kind="div"):
        t, _ = steps[name]
        if t.dim() == 3:                               # (H, T, T): one matrix per head
            return [pack(h, kind) for h in t]
        return pack(t, kind)

    data = dict(
        seed=args.seed,
        dims=dict(T=144, D=384, heads=6, d_head=64, hidden=1536, blocks=12),
        params=sum(p.numel() for p in model.parameters()),
        macs={name: macs for name, _, macs in trace},
        x=pack(x),
        ln1=step("LN1"), ln1_gamma=pack(blk.ln1.weight), ln1_beta=pack(blk.ln1.bias),
        w_qkv=pack(blk.qkv.weight.T), b_qkv=pack(blk.qkv.bias), qkv=step("QKV projection"),
        scores=step("scores Q.K^T / sqrt(d)"), probs=step("softmax", "seq"),
        heads_out=step("P.V"),
        w_o=pack(blk.proj.weight.T), b_o=pack(blk.proj.bias), proj=step("output projection"),
        x1=step("residual 1"),
        ln2=step("LN2"), ln2_gamma=pack(blk.ln2.weight), ln2_beta=pack(blk.ln2.bias),
        w_fc1=pack(blk.fc1.weight.T), b_fc1=pack(blk.fc1.bias), fc1=step("FC1"),
        gelu=step("GELU"),
        w_fc2=pack(blk.fc2.weight.T), b_fc2=pack(blk.fc2.bias), fc2=step("FC2"),
        block_out=outs,
    )
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
