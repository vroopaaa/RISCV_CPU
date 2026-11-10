#!/usr/bin/env python3
"""Plain PyTorch model of Sparse-ViT-Det's encoder (../vit_block.md): ViT-Small
blocks run on the 144 sampled tokens. Float32, random weights.

Prints every step of one block (output shape and multiply-accumulates), then runs
the full 12-block encoder.

Usage:
  python3 vit_block_torch.py
  python3 vit_block_torch.py --tokens 144 --blocks 12 --seed 0
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBlock(nn.Module):
    """Pre-norm transformer encoder block:
         x = x + proj(attention(ln1(x)))
         x = x + fc2(gelu(fc1(ln2(x))))
    """

    def __init__(self, dim=384, heads=6, mlp_hidden=1536):
        super().__init__()
        self.heads = heads
        self.d_head = dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)

    def forward(self, x, trace=None):
        """x: (T, D). trace: an optional list that gets (step, tensor, MACs) per step."""
        T, D = x.shape
        H, d = self.heads, self.d_head

        def log(step, t, macs=0):
            if trace is not None:
                trace.append((step, t, macs))
            return t

        h = log("LN1", self.ln1(x))
        qkv = log("QKV projection", self.qkv(h), T * D * 3 * D)
        q, k, v = qkv.split(D, dim=1)
        q = q.view(T, H, d).transpose(0, 1)                  # (H, T, d)
        k = k.view(T, H, d).transpose(0, 1)
        v = v.view(T, H, d).transpose(0, 1)
        s = log("scores Q.K^T / sqrt(d)", q @ k.transpose(1, 2) / math.sqrt(d), H * T * d * T)
        p = log("softmax", s.softmax(dim=-1))
        o = log("P.V", (p @ v).transpose(0, 1).reshape(T, D), H * T * T * d)
        a = log("output projection", self.proj(o), T * D * D)
        x = log("residual 1", x + a)
        h = log("LN2", self.ln2(x))
        h = log("FC1", self.fc1(h), T * D * self.fc1.out_features)
        h = log("GELU", F.gelu(h))
        h = log("FC2", self.fc2(h), T * self.fc1.out_features * D)
        x = log("residual 2", x + h)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, blocks=12, dim=384, heads=6, mlp_hidden=1536):
        super().__init__()
        self.blocks = nn.ModuleList(ViTBlock(dim, heads, mlp_hidden) for _ in range(blocks))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def main():
    ap = argparse.ArgumentParser(description="Run Sparse-ViT-Det's ViT encoder in PyTorch.")
    ap.add_argument("--tokens", type=int, default=144)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = ViTEncoder(blocks=args.blocks).eval()
    x = torch.randn(args.tokens, 384)                       # the sampled tokens, (T, D)

    # One block, step by step.
    trace = []
    with torch.no_grad():
        model.blocks[0](x, trace)
    print(f"One block, T = {args.tokens}, D = 384, 6 heads x 64, MLP 1536\n")
    print(f"{'step':<24} {'output shape':<18} {'MACs':>13}")
    total = 0
    for step, t, macs in trace:
        total += macs
        print(f"{step:<24} {str(tuple(t.shape)):<18} {macs:>13,}")
    print(f"{'total':<24} {'':<18} {total:>13,}")

    # The whole encoder.
    with torch.no_grad():
        y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nEncoder: {args.blocks} blocks, {params:,} parameters, "
          f"{args.blocks * total:,} MACs")
    print(f"input  {tuple(x.shape)}  mean {x.mean():.4f}  std {x.std():.4f}")
    print(f"output {tuple(y.shape)}  mean {y.mean():.4f}  std {y.std():.4f}")


if __name__ == "__main__":
    main()
