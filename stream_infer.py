"""Streaming (frame-by-frame) inference example for S-Gate.

Demonstrates:
  * win_length input ring buffer
  * per-hop GRU/Conv state carry
  * overlap-add reconstruction of the output stream
"""

from __future__ import annotations

import argparse

import torch

from model import SGateModel


@torch.no_grad()
def stream_enhance(model: SGateModel, wav: torch.Tensor) -> torch.Tensor:
    """wav: [T]  ->  enhanced [T] (mono, single example)."""
    model.eval()
    device = next(model.parameters()).device
    wav = wav.to(device)
    T = wav.shape[-1]
    win, hop = model.win_length, model.hop_length

    state = model.init_state(batch_size=1, device=device)
    in_buf  = torch.zeros(1, win, device=device)
    out_buf = torch.zeros(1, T + win, device=device)

    n_frames = max(0, (T - win) // hop + 1)
    for i in range(n_frames):
        start = i * hop
        in_buf[0] = wav[start:start + win]
        out_hop, state = model.stream_step(in_buf, state)
        # Overlap-add: place the new hop at its time position.
        out_buf[0, start + win - hop:start + win] += out_hop[0]

    return out_buf[0, :T].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="Optional model checkpoint (.pt).")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    model = SGateModel()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    # Synthetic noisy signal for the demo
    import math
    n = int(args.seconds * args.sr)
    t = torch.arange(n) / args.sr
    clean = 0.4 * torch.sin(2 * math.pi * 440.0 * t)
    noisy = clean + 0.2 * torch.randn(n)

    enhanced = stream_enhance(model, noisy)
    print(f"in: {noisy.shape}  out: {enhanced.shape}")
    print(f"input  RMS: {noisy.pow(2).mean().sqrt():.4f}")
    print(f"output RMS: {enhanced.pow(2).mean().sqrt():.4f}")


if __name__ == "__main__":
    main()
