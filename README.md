# S-Gate (Structure-Gate Network)

Lightweight, causal speech enhancement baseline.

## Files
- [model.py](model.py) — `SGateModel`, STFT helpers, spectral flux, streaming step.
- [losses.py](losses.py) — SI-SNR + multi-resolution STFT (+ optional perceptual weight).
- [train.py](train.py) — minimal training loop with a dummy dataset.
- [stream_infer.py](stream_infer.py) — frame-by-frame streaming inference example.

## Design at a glance
- STFT: `n_fft=256`, `hop=80` (5 ms @ 16 kHz), `win=160` (10 ms), `center=False` → causal.
- Features: magnitude + spectral flux (causal frame difference).
- Encoder: 2× depthwise-separable Conv2D, causal time padding, freq stride 2 (129 → 33).
- Core: single-layer GRU (hidden 128).
- Mask head: 2× Linear + sigmoid → per-bin gain in [0, 1].
- Reconstruction: `mask * |X|` with original phase, iSTFT, `tanh`.

## Constraints satisfied
- Only Conv/GRU/Linear/Sigmoid/tanh/BN — INT8-friendly, no transformers/SSM/IIR.
- Strictly causal (no future frames; streaming step verified).
- < 500K parameters with the default config (run `python model.py` to print the count).
- ≤ 10 ms algorithmic latency (one analysis window).

## Quick start
```bash
python model.py            # print param count + run a forward pass
python train.py --epochs 1 # sanity-check training on dummy data
python stream_infer.py     # streaming demo
```
