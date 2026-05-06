"""Colab feasibility check for S-Gate training.

Run this BEFORE kicking off a long training job. It prints:
  1. GPU name + total / available memory
  2. Local disk free space + Google Drive free space (if mounted)
  3. Largest batch size that fits in VRAM (from [8, 16, 32, 64])
  4. Per-batch latency over 10 forward+backward steps -> epoch time estimate
  5. A green/red summary with warnings

Usage (Colab):
    !python check_colab_feasibility.py --epoch-batches 1024
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# 1. GPU
# ---------------------------------------------------------------------------

def gpu_info():
    if not torch.cuda.is_available():
        print("GPU            : <none> (CPU only)")
        return None
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU            : {name}")
    print(f"  total memory : {total/1e9:6.2f} GB")
    print(f"  free  memory : {free/1e9:6.2f} GB")
    return {"name": name, "total": total, "free": free}


# ---------------------------------------------------------------------------
# 2. Disk
# ---------------------------------------------------------------------------

def disk_info():
    info = {}
    local = shutil.disk_usage("/content" if os.path.isdir("/content") else "/")
    print(f"Local disk     : {local.free/1e9:6.2f} GB free / {local.total/1e9:6.2f} GB total")
    info["local_free_gb"] = local.free / 1e9
    drive_path = "/content/drive/MyDrive"
    if os.path.isdir(drive_path):
        d = shutil.disk_usage(drive_path)
        print(f"Google Drive   : {d.free/1e9:6.2f} GB free / {d.total/1e9:6.2f} GB total")
        info["drive_free_gb"] = d.free / 1e9
    else:
        print("Google Drive   : not mounted (/content/drive/MyDrive missing)")
        info["drive_free_gb"] = None
    return info


# ---------------------------------------------------------------------------
# Model loader (uses repo SGateModel if present, else a tiny stand-in)
# ---------------------------------------------------------------------------

def build_model(device):
    try:
        from model import SGateModel, count_parameters
        m = SGateModel().to(device)
        n = count_parameters(m)
        print(f"Model          : SGateModel  ({n:,} params)")
        return m
    except Exception as e:
        print(f"Model          : repo SGateModel not importable ({e}); using stub")
        import torch.nn as nn
        class Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(129, 128, batch_first=True)
                self.head = nn.Linear(128, 129)
            def forward(self, wav):
                spec = torch.stft(wav, n_fft=256, hop_length=80, win_length=160,
                                  window=torch.hann_window(160, device=wav.device),
                                  center=False, return_complex=True)
                mag = spec.abs().transpose(1, 2)
                h, _ = self.gru(mag)
                m = torch.sigmoid(self.head(h)).transpose(1, 2)
                return torch.istft(spec * m, n_fft=256, hop_length=80, win_length=160,
                                   window=torch.hann_window(160, device=wav.device),
                                   center=False, length=wav.shape[-1])
        return Stub().to(device)


# ---------------------------------------------------------------------------
# 3. Batch size sweep
# ---------------------------------------------------------------------------

def try_batch(model, batch: int, n_samples: int, device, use_amp: bool) -> bool:
    try:
        torch.cuda.empty_cache() if device.type == "cuda" else None
        x = torch.randn(batch, n_samples, device=device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            y = model(x)
            loss = y.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise


def batch_sweep(model, n_samples: int, device, candidates: List[int], use_amp: bool) -> int:
    print("\nBatch size test:")
    best = 0
    for bs in candidates:
        ok = try_batch(model, bs, n_samples, device, use_amp)
        print(f"  batch={bs:>3d}  {'OK' if ok else 'OOM'}")
        if ok:
            best = bs
        else:
            break
    return best


# ---------------------------------------------------------------------------
# 4. Time estimation
# ---------------------------------------------------------------------------

def time_estimate(model, batch: int, n_samples: int, device, use_amp: bool,
                  n_iters: int = 10):
    if batch == 0:
        return None
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # warmup
    x = torch.randn(batch, n_samples, device=device)
    for _ in range(3):
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(x).float().pow(2).mean()
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        optim.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(x).float().pow(2).mean()
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        optim.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iters
    return dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seg-seconds", type=float, default=2.0)
    ap.add_argument("--candidates", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--epoch-batches", type=int, default=1024,
                    help="Number of batches per epoch (used for ETA).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    n_samples = int(args.sample_rate * args.seg_seconds)
    print("=" * 60)
    print("S-Gate Colab feasibility check")
    print("=" * 60)
    print(f"Segment        : {args.seg_seconds}s @ {args.sample_rate} Hz = {n_samples} samples")
    print(f"AMP            : {'OFF' if args.no_amp else 'ON (fp16 autocast)'}\n")

    g = gpu_info()
    d = disk_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    model = build_model(device)
    model.train()

    best_bs = batch_sweep(model, n_samples, device, args.candidates, use_amp)

    per_batch = time_estimate(model, best_bs, n_samples, device, use_amp) if best_bs else None
    if per_batch is not None:
        epoch_s = per_batch * args.epoch_batches
        total_s = epoch_s * args.epochs
        print(f"\nTiming (batch={best_bs}):")
        print(f"  per-batch   : {per_batch*1000:7.1f} ms")
        print(f"  per-epoch   : {epoch_s/60:7.2f} min  ({args.epoch_batches} batches)")
        print(f"  full run    : {total_s/3600:7.2f} h    ({args.epochs} epochs)")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    warnings = []
    if g is None:
        warnings.append("No GPU detected. Training will be very slow on CPU.")
    elif g["free"] < 2e9:
        warnings.append(f"Low free GPU memory ({g['free']/1e9:.2f} GB). "
                        f"Restart runtime to reclaim VRAM.")
    if d.get("drive_free_gb") is not None and d["drive_free_gb"] < 5:
        warnings.append(f"Google Drive < 5 GB free; checkpoints may fail to save.")
    if d["local_free_gb"] < 5:
        warnings.append(f"Local /content disk < 5 GB free; dataset extraction may fail.")
    if best_bs == 0:
        warnings.append("Even batch=8 OOMed -> reduce segment length or model width.")

    if best_bs:
        print(f"  [OK] Safe batch size : {best_bs}")
    if per_batch is not None:
        print(f"  [OK] Epoch time est. : {per_batch*args.epoch_batches/60:.2f} min")
        print(f"  [OK] Full run est.   : {per_batch*args.epoch_batches*args.epochs/3600:.2f} h")
    for w in warnings:
        print(f"  [!]  {w}")
    if not warnings:
        print("  [OK] No resource warnings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
