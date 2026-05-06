# S-Gate — Project Context (for low-token debugging / upgrades)

This file is a compact, self-contained snapshot of the project so any future
chat session can pick it up without re-reading every source file.

---

## 1. What this project is

**S-Gate (Structure-Gate Network)** — lightweight, causal, real-time speech
enhancement model.

Goals:
- Enhance speech in noise; preserve music / birds / environmental sounds.
- Suppress unstructured noise (traffic, hum).
- < 10 ms algorithmic latency, < 500K params, INT8-friendly.
- Deployable on mobile / DSP.

Allowed components only: STFT, Conv1D/2D, single-layer GRU, Linear,
Sigmoid/tanh, BatchNorm. **No** transformers, SSM/Mamba, IIR, GANs,
phase derivatives, perceptual nets.

---

## 2. Repo layout

```
sgate/
├── model.py                    # SGateModel + STFT helpers + streaming
├── losses.py                   # SI-SNR + freq-weighted MR-STFT + TV
├── train.py                    # AdamW + warmup/cosine + AMP + finetune
├── stream_infer.py             # frame-by-frame demo with overlap-add
├── colab_train.ipynb           # full Colab pipeline (Drive + GitHub)
├── check_colab_feasibility.py  # GPU/disk/batch/time pre-check
├── README.md
└── CONTEXT.md                  # this file
```

---

## 3. Architecture (SGateModel) — ~160K params

```
wav [B, T]
  │
  ▼ STFT (n_fft=256, hop=80 [5ms], win=160 [10ms], center=False, causal)
mag, phase  [B, F=129, T_frames]
  │
  ├── flux = mag[t] - mag[t-1]   (causal, zero-pad first)
  ▼
[B, 2, T, F]
  │
  ▼ enc1: depthwise-separable Conv2D, k=(3 t, 5 f), causal time pad, freq stride 2
[B, 16, T, 65]
  │
  ▼ enc2: same, freq stride 2
[B, 24, T, 33]
  │
  ▼ flatten freq -> Linear(24*33 -> 64)
[B, T, 64]
  │
  ▼ GRU (1 layer, hidden=128, batch_first)
[B, T, 128]
  │
  ▼ Linear 128->128, ReLU, Linear 128->129, Sigmoid
mask [B, T, 129] -> transpose -> [B, F, T]
  │
  ▼ Mask smoother: Conv2d(1,1,3x3), causal time, identity-init  (9 params)
  ▼ mask_floor: mask = 0.05 + 0.95 * mask    (anti over-suppression)
  │
enh_mag = mag * mask
  ▼ iSTFT(orig phase) -> tanh
enhanced wav [B, T]
```

Key constants (defaults):
- `n_fft=256`, `hop=80`, `win=160`, `n_freq=129`, `freq_after_enc=33`
- `enc_channels=(16, 24)`, `freq_reduced=64`, `gru_hidden=128`
- `mask_floor=0.05`, `smooth_mask=True`

Streaming (`stream_step`) caches: `gru.h`, `prev_mag`, `enc1_hist` (2 frames),
`enc2_hist` (2 frames), `mask_hist` (2 frames). Algorithmic latency = win = 10 ms.

---

## 4. Loss (SGateLoss) — formula

$$L = w_{si}\cdot\mathrm{clamp}(-\text{SI-SNR},\ \text{max}=30) + w_{mr}\cdot L_{\text{MR-STFT}}^{w(f)} \;[+ w_p\cdot L_{\text{MR-STFT}}^{\text{speech-band}}]\;[+ w_{tv}\cdot L_{TV}(\text{mask})]$$

- SI-SNR clamped at 30 dB (prevents gradient blow-up).
- MR-STFT: spectral_convergence + log_mag, **frequency-weighted** by a
  raised-cosine bump on 1–4 kHz (boost=2 over base=1).
  Resolutions: `n_fft=(512,1024,2048)`, `hop=(50,120,240)`, `win=(240,600,1200)`.
- Perceptual term = same MR-STFT applied to a 300–3400 Hz STFT-band-passed
  waveform.
- TV penalty: `mean|m_t - m_{t-1}| + 0.5 * mean|m_f - m_{f-1}|`.

Default weights — base run: `w_sisnr=0.5, w_mrstft=1.0, w_perc=0.5,
w_mask_tv=0` (perceptual ON after epoch 8).
Fine-tune: `w_sisnr=0.3, w_mrstft=1.0, w_perc=1.0, w_mask_tv=0.05`,
`si_snr_max_db=25`.

`SGateLoss.forward(est, ref, mask=None)` returns `(loss_tensor, components_dict)`.
Components dict keys: `total, sisnr, mrstft, [perc], [mask_tv]`.

---

## 5. Training (train.py)

`TrainConfig` dataclass holds everything. Key defaults:

| field | base | finetune |
|---|---|---|
| epochs | 60 | ≥10 |
| batch_size | 128 | 128 |
| lr_start / lr_max / lr_min | 1e-5 / 3e-4 / 1e-6 | 1e-6 / 5e-5 / 1e-6 |
| warmup_steps | 1000 | 200 |
| weight_decay | 1e-4 | 1e-4 |
| grad_clip | 1.0 | 1.0 |
| optimizer | AdamW | AdamW |
| AMP | on (CUDA) | on (CUDA) |
| snr_range_db | (-5, 20) | (-5, 20) |
| peak_normalize | True | True |

LR schedule: linear warmup → cosine decay (`lr_at_step()` set per step).
Augmentation: random per-step SNR mix (`mix_at_snr`), then peak-normalize
noisy with same gain on clean (preserves SNR).
NaN/Inf guard: skip step, increment `skipped` counter.
Checkpoint: every epoch -> `sgate_epoch{NN}.pt` containing
`{model, optim, scaler, epoch, step, cfg}`.

CLI:
```bash
python train.py --epochs 60                                   # base
python train.py --finetune --init-from sgate_epoch59.pt --epochs 10
```

---

## 6. Streaming inference (stream_infer.py / model.stream_step)

- Caller maintains a `win_length`-sample input ring buffer.
- Each `stream_step(frame_wav, state)` consumes 1 win and emits `hop_length`
  output samples (caller does overlap-add for the prior `win - hop` samples).
- State is a dict; allocate via `model.init_state(batch_size, device)`.

---

## 7. Colab pipeline (colab_train.ipynb)

10 cells, in order:
1. Install librosa/pesq/pystoi/soundfile; print device.
2. Mount Drive; create `data/`, `checkpoints/`, `logs/` under
   `/content/drive/MyDrive/sgate/`.
3. Optional ZIP upload + `git push` to GitHub (token-in-URL).
4. `git clone` fresh into `/content/sgate`; add to `sys.path`.
5. LibriSpeech (.flac) + MUSAN (.wav) loaders. File lists cached to Drive.
   Sub-sampled (4000 speech, 1000 noise) by default.
6. Auto batch-size (`find_max_batch`, doubles until OOM, uses half).
   AMP, grad_clip=1, num_workers=2, pin_memory=True, persistent_workers=True.
7. Training loop with warmup+cosine LR, NaN skip, it/s + ETA logging,
   epoch checkpoints to Drive, log mirrored to `logs/train.log`.
8. Auto-resume from latest `sgate_epoch*.pt` on Drive.
9. Validation: SI-SNR + PESQ (wb) + STOI on a small held-out batch.
10. Inference demo -> writes clean/noisy/enhanced WAV to Drive + inline play.

User must edit `GITHUB_USER`, `GITHUB_REPO`, `GITHUB_TOKEN`,
`GIT_USER_EMAIL/NAME` in cell 3 once.

---

## 8. Feasibility check (check_colab_feasibility.py)

Standalone script. Reports GPU mem, local + Drive free space, sweeps batch
sizes [8, 16, 32, 64] for the largest that fits, then runs 10 backward steps
to estimate per-batch / per-epoch / full-run wall time. Prints a summary with
warnings if resources are insufficient. Uses repo `SGateModel` if importable,
otherwise a tiny stub with the same I/O shape.

---

## 9. Public APIs (signatures only)

```python
# model.py
class SGateModel(nn.Module):
    def __init__(self, n_fft=256, hop_length=80, win_length=160,
                 enc_channels=(16,24), freq_reduced=64, gru_hidden=128,
                 mask_floor=0.05, smooth_mask=True, debug=False): ...
    def forward(self, wav, return_mask=False): ...
    def init_state(self, batch_size=1, device=None) -> dict: ...
    def stream_step(self, frame_wav, state) -> (out_hop, state): ...

class STFT(nn.Module):
    def stft(self, wav)  -> (mag, phase): ...
    def istft(self, mag, phase, length=None) -> wav: ...

def spectral_flux(mag) -> mag: ...
def count_parameters(model) -> int: ...

# losses.py
def si_snr(est, ref) -> [B] dB
def si_snr_loss(est, ref, max_db=30.0) -> scalar
class STFTLoss / MultiResSTFTLoss(freq_emphasis=True, sample_rate=16000)
class SGateLoss:
    def __init__(self, w_sisnr=0.5, w_mrstft=1.0, w_perc=0.5,
                 w_mask_tv=0.0, si_snr_max_db=30.0,
                 perceptual=False, freq_emphasis=True, sample_rate=16000)
    def forward(self, est, ref, mask=None) -> (loss, components_dict)
    def set_perceptual(self, on: bool)

# train.py
@dataclass class TrainConfig: ...     # see §5
def train(cfg: TrainConfig): ...
def lr_at_step(step, total, cfg) -> float
def mix_at_snr(clean, noise, snr_db) -> noisy
```

---

## 10. Common pitfalls (lessons learned)

- **Center=False** in STFT is required for causality. `torch.istft` with
  `center=False` returns slightly fewer samples — always trim/pad before SI-SNR.
- SI-SNR in dB is unbounded -> **must** clamp (we use 30 dB).
- LR > 5e-4 with this size GRU diverges within a few hundred steps.
- `grad_clip > 5` lets occasional STFT spikes leak through.
- BatchNorm is fine in causal Conv2D because it normalises over `(B, T*F')`,
  not across time — INT8 fold works.
- Mask smoother is **identity-initialised** (kernel `[0,0,0; 0,0,0; 0,1,0]`)
  so it doesn't damage early training; it learns a small averaging stencil.
- On Colab, load file lists from a cached `.txt` — `glob('**/*.flac')` over
  Drive takes minutes.

---

## 11. Quick commands

```bash
# Local sanity
python model.py                       # forward pass + param count
python train.py --epochs 1            # 1-epoch dummy training
python stream_infer.py                # streaming demo

# Fine-tune (after base converges)
python train.py --finetune --init-from sgate_epoch59.pt --epochs 10

# Colab pre-flight
python check_colab_feasibility.py --epoch-batches 1024 --epochs 30
```

---

## 12. What is NOT in scope

- Real perceptual networks (PESQNet, WavLM losses) — explicitly forbidden.
- GAN-based losses, adversarial training.
- Multi-channel / array processing.
- Sample-rate other than 16 kHz (would need re-tuned freq weights).
- IIR or learned phase estimation.
