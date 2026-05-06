"""Training pipeline for S-Gate (real-time speech enhancement).

Improvements over v1:

  * AdamW + linear warmup -> cosine decay LR schedule (per-step)
  * Bounded SI-SNR + balanced MR-STFT (see losses.py)
  * AMP (mixed precision) on CUDA, with grad scaler
  * Strict gradient clipping (max_norm=1.0) and global-norm logging
  * NaN / Inf guard: skip the step instead of crashing
  * On-the-fly random SNR mixing in [-5, 20] dB and waveform peak-normalization
  * Warm-start: spectral-only loss for the first warmup epochs, then enable
    the perceptual (speech-band) term
  * Per-component loss logging
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from losses import SGateLoss, si_snr
from model import SGateModel, count_parameters


# Bump these whenever loss/architecture changes — `safe_load` uses them to
# refuse a `full` resume across incompatible runs.
ARCH_VERSION = 3
LOSS_VERSION = 4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    snr_range_db: Tuple[float, float] = (0.0, 25.0)   # real-world distribution
    peak_normalize: bool = True
    reverb_prob: float = 0.4              # probability of adding synthetic reverb
    reverb_rt60_range: Tuple[float, float] = (0.1, 0.5)  # seconds

    # Optimization
    epochs: int = 60
    batch_size: int = 128
    lr_start: float = 1e-5
    lr_max:   float = 2e-4            # reduced from 3e-4 (was unstable)
    lr_min:   float = 1e-6
    warmup_steps: int = 1500          # longer warmup stabilizes complex mask
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Loss (aligned with new SGateLoss defaults)
    w_sisnr: float = 0.3
    w_mrstft: float = 0.7
    w_mel: float = 0.5
    w_complex: float = 1.0           # PHASE-AWARE — main PESQ driver
    w_perc: float = 0.3
    w_mask_tv: float = 0.02
    si_snr_max_db: float = 30.0
    perceptual_start_epoch: int = 5   # earlier start for mel+perceptual
    freq_emphasis: bool = True

    # Fine-tune phase (set --finetune)
    finetune: bool = False            # if True, use perceptual+TV preset and lower LR

    # Misc
    amp: bool = True                  # mixed precision on CUDA
    workers: int = 4
    log_every: int = 50

    # Checkpoint / resume strategy
    ckpt_dir: str = '.'
    save_best_only: bool = True       # only persist new best.pt; cheap last.pt every N epochs
    save_last_every: int = 2          # 0 disables
    resume_mode: str = 'warm'         # 'full' | 'warm' | 'partial' | 'none'
    resume_from: Optional[str] = None
    min_resume_sisnr_db: float = 5.0  # below this -> downgrade full -> warm

    # Plateau / early stopping
    patience: int = 3                 # epochs without improvement -> warm restart
    min_delta_db: float = 0.1
    hard_stop_patience: int = 6       # plateaus this many epochs -> stop
    warm_restart_lr_mult: float = 10.0
    warm_restart_lr_cap: float = 3e-4


# ---------------------------------------------------------------------------
# Dataset (dummy structure – replace with your real loader)
# ---------------------------------------------------------------------------

class DummyEnhanceDataset(Dataset):
    """Returns (clean, noise) pairs. Mixing is done in the train loop so SNR
    can be sampled per step (a stronger augmentation than baking it into __getitem__)."""

    def __init__(self, num_items: int = 4096, sample_rate: int = 16000, dur_s: float = 2.0):
        self.num_items = num_items
        self.sr = sample_rate
        self.n = int(sample_rate * dur_s)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int):
        t = torch.arange(self.n) / self.sr
        f1 = 200 + 600 * torch.rand(1).item()
        f2 = 400 + 1200 * torch.rand(1).item()
        clean = 0.4 * torch.sin(2 * math.pi * f1 * t) + 0.3 * torch.sin(2 * math.pi * f2 * t)
        white = torch.randn(self.n)
        kernel = torch.tensor([0.25, 0.5, 0.25])
        colored = F.conv1d(white.view(1, 1, -1), kernel.view(1, 1, -1), padding=1).view(-1)
        noise = colored if torch.rand(1).item() < 0.5 else white
        return clean.float(), noise.float()


# ---------------------------------------------------------------------------
# Augmentation: random SNR mixing
# ---------------------------------------------------------------------------

def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """clean, noise: [B, T]; snr_db: [B]. Scales noise to hit the target SNR."""
    cp = clean.pow(2).mean(dim=-1, keepdim=True) + eps
    np_ = noise.pow(2).mean(dim=-1, keepdim=True) + eps
    target_np = cp / (10.0 ** (snr_db.view(-1, 1) / 10.0))
    noise = noise * (target_np / np_).sqrt()
    return clean + noise


# ---------------------------------------------------------------------------
# Augmentation: synthetic room reverb (improves real-world generalization)
# ---------------------------------------------------------------------------

def add_reverb_batch(clean: torch.Tensor, rt60_range: Tuple[float, float],
                     sr: int = 16000, prob: float = 0.4) -> torch.Tensor:
    """Add exponential-decay reverb to a random subset of the batch.

    clean: [B, T]. Returns reverberant clean (same shape).
    No external libraries needed — uses a simple synthetic IR.
    """
    B, T = clean.shape
    device = clean.device
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return clean
    # Generate per-sample RT60
    rt60 = torch.empty(B, device=device).uniform_(*rt60_range)
    max_len = int(rt60_range[1] * sr)
    # Exponential decay IR
    t_axis = torch.arange(max_len, device=device).float() / sr     # [L]
    decay = torch.exp(-6.9 / rt60.unsqueeze(1) * t_axis.unsqueeze(0))  # [B, L]
    ir = torch.randn(B, max_len, device=device) * decay
    # Normalize IR energy
    ir = ir / ir.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    ir[0, 0] = 1.0  # direct path
    # Convolve using F.conv1d (per-sample via groups)
    out = clean.clone()
    for i in range(B):
        if mask[i]:
            reverbed = F.conv1d(
                clean[i:i+1].unsqueeze(0),
                ir[i:i+1].unsqueeze(0),
                padding=max_len // 2,
            ).squeeze(0).squeeze(0)[:T]
            out[i] = reverbed
    return out


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------

def lr_at_step(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr_start + (cfg.lr_max - cfg.lr_start) * (step / max(1, cfg.warmup_steps))
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


# ---------------------------------------------------------------------------
# Resume strategy
# ---------------------------------------------------------------------------

def safe_load(model, optim, scaler, path: Optional[str], mode: str,
              device, min_resume_sisnr: float):
    """Load a checkpoint with the requested strategy.

    Modes:
      full     -> model + optimizer + scaler + epoch + best (strict load)
      warm     -> model weights only; optimizer & schedule fresh; best=-inf
      partial  -> model with strict=False (handles arch changes); fresh opt
      none     -> nothing loaded

    Safety gates:
      * full mode is downgraded to warm if checkpoint best SI-SNR is below
        ``min_resume_sisnr`` (avoids locking into a bad basin).
      * full mode is downgraded to partial if arch/loss version tags differ.

    Returns: (start_epoch, best_metric, effective_mode)
    """
    if mode == 'none' or not path or not os.path.exists(path):
        print(f'[resume] starting from scratch (mode={mode})')
        return 0, float('-inf'), 'none'

    ckpt = torch.load(path, map_location=device)
    prev_best = float(ckpt.get('best', float('-inf')))
    cfg_tag = ckpt.get('cfg_tag', {}) or {}
    arch_v = cfg_tag.get('arch_version')
    loss_v = cfg_tag.get('loss_version')

    # Refuse full resume on weak checkpoints.
    if mode == 'full' and prev_best < min_resume_sisnr:
        print(f'[resume] ckpt best={prev_best:.2f} dB < {min_resume_sisnr}; '
              f'downgrading FULL -> WARM')
        mode = 'warm'
    # Refuse full resume across incompatible code versions.
    if mode == 'full' and (arch_v != ARCH_VERSION or loss_v != LOSS_VERSION):
        print(f'[resume] cfg tag mismatch (ckpt arch={arch_v}, loss={loss_v} '
              f'vs current arch={ARCH_VERSION}, loss={LOSS_VERSION}); '
              f'downgrading FULL -> PARTIAL')
        mode = 'partial'

    state = ckpt.get('model', ckpt)

    if mode == 'full':
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f'[resume] strict load failed ({str(e).splitlines()[0]}); '
                  f'downgrading FULL -> PARTIAL')
            mode = 'partial'

    if mode != 'full':
        # Filter incompatible tensors (handles arch shape changes).
        model_state = model.state_dict()
        compatible, skipped = {}, []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            else:
                reason = ('shape ' + str(tuple(v.shape)) + ' vs '
                          + str(tuple(model_state[k].shape))) if k in model_state \
                         else 'not in current model'
                skipped.append((k, reason))
        if skipped:
            print(f'[resume] skipping {len(skipped)} incompatible tensor(s):')
            for k, r in skipped[:5]:
                print(f'           - {k}  ({r})')
            if len(skipped) > 5:
                print(f'           ... and {len(skipped)-5} more')
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if missing:
            print(f'[resume] missing keys:    {len(missing)} (e.g. {missing[:3]})')
        if unexpected:
            print(f'[resume] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})')

    if mode == 'full':
        if 'optim' in ckpt: optim.load_state_dict(ckpt['optim'])
        if scaler is not None and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best = prev_best
        print(f'[resume] FULL from epoch {start_epoch}, best={best:+.3f} dB')
    else:
        # Force best to -inf so any new epoch can win and overwrite best.pt.
        # Optimizer momentum is reset (any leftover Adam moments are poisoned).
        start_epoch = 0
        best = float('-inf')
        for st in optim.state.values():
            st.clear()
        print(f'[resume] {mode.upper()} start: weights loaded, optimizer reset')

    return start_epoch, best, mode


# ---------------------------------------------------------------------------
# Validation (clean -> SI-SNR @ fixed 5 dB SNR)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_sisnr(model, loader, device, max_batches: int = 8) -> float:
    model.eval()
    vals = []
    for i, (clean, noise) in enumerate(loader):
        if i >= max_batches:
            break
        clean = clean.to(device); noise = noise.to(device)
        snr = torch.full((clean.size(0),), 5.0, device=device)
        noisy = mix_at_snr(clean, noise, snr)
        est = model(noisy)
        L = min(est.shape[-1], clean.shape[-1])
        vals.append(float(si_snr(est[..., :L].float(), clean[..., :L].float()).mean()))
    model.train()
    return float(sum(vals) / max(1, len(vals)))


# ---------------------------------------------------------------------------
# Plateau handling: warm restart (LR boost + wipe optimizer moments)
# ---------------------------------------------------------------------------

def warm_restart(optim, lr_mult: float, lr_cap: float) -> None:
    for g in optim.param_groups:
        g['lr'] = min(g['lr'] * lr_mult, lr_cap)
    for st in optim.state.values():
        st.clear()
    print(f'[lr] plateau -> warm restart (LR x{lr_mult}, fresh momentum)')


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    print(f"device: {device}  amp: {use_amp}")

    model = SGateModel().to(device)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,}")
    assert n_params < 500_000, "Model exceeds 500K parameter budget"

    loss_fn = SGateLoss(
        w_sisnr=cfg.w_sisnr, w_mrstft=cfg.w_mrstft, w_mel=cfg.w_mel,
        w_complex=cfg.w_complex, w_perc=cfg.w_perc, w_mask_tv=cfg.w_mask_tv,
        si_snr_max_db=cfg.si_snr_max_db, perceptual=cfg.finetune,
        freq_emphasis=cfg.freq_emphasis,
        sample_rate=cfg.sample_rate,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr_start,
                              betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_ds = DummyEnhanceDataset(num_items=cfg.batch_size * 64,
                                   sample_rate=cfg.sample_rate,
                                   dur_s=cfg.segment_seconds)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.workers, drop_last=True, pin_memory=True)
    val_ds = DummyEnhanceDataset(num_items=cfg.batch_size * 4,
                                 sample_rate=cfg.sample_rate,
                                 dur_s=cfg.segment_seconds)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=max(0, cfg.workers // 2), pin_memory=True)

    # ---- Resume strategy (safe_load handles full / warm / partial / none) ----
    start_epoch, best_sisnr, eff_mode = safe_load(
        model, optim, scaler if use_amp else None,
        cfg.resume_from, cfg.resume_mode, device, cfg.min_resume_sisnr_db,
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_path = os.path.join(cfg.ckpt_dir, 'sgate_best.pt')
    last_path = os.path.join(cfg.ckpt_dir, 'sgate_last.pt')
    cfg_tag = {'arch_version': ARCH_VERSION, 'loss_version': LOSS_VERSION}

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.epochs
    snr_lo, snr_hi = cfg.snr_range_db

    global_step = 0
    skipped = 0
    plateau = 0
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        # Warm-start: turn on perceptual term once spectral target is well-fit.
        # In fine-tune mode it stays on for the entire run.
        loss_fn.perceptual = cfg.finetune or (epoch >= cfg.perceptual_start_epoch)

        t0 = time.time()
        running = {"total": 0.0, "sisnr": 0.0, "mrstft": 0.0, "perc": 0.0,
                   "gnorm": 0.0, "n": 0}
        for clean, noise in loader:
            clean = clean.to(device, non_blocking=True)
            noise = noise.to(device, non_blocking=True)

            # ---- on-the-fly augmentation ----
            # Synthetic reverb on clean (before mixing) to simulate rooms.
            if cfg.reverb_prob > 0:
                clean = add_reverb_batch(clean, cfg.reverb_rt60_range,
                                         sr=cfg.sample_rate, prob=cfg.reverb_prob)
            snr = torch.empty(clean.size(0), device=device).uniform_(snr_lo, snr_hi)
            noisy = mix_at_snr(clean, noise, snr)
            if cfg.peak_normalize:
                # Normalize noisy and apply same gain to clean (preserves SNR).
                peak = noisy.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                gain = (0.95 / peak).clamp(max=1.0)
                noisy = noisy * gain
                clean = clean * gain

            # ---- LR schedule (per step) ----
            lr = lr_at_step(global_step, total_steps, cfg)
            set_lr(optim, lr)

            # ---- forward + loss (AMP) ----
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                est, mask = model(noisy, return_mask=True)
                L = min(est.shape[-1], clean.shape[-1])
                loss, comps = loss_fn(est[..., :L], clean[..., :L], mask=mask)

            # ---- NaN / Inf guard ----
            if has_nan_or_inf(loss):
                skipped += 1
                print(f"[step {global_step}] non-finite loss, skipping. comps={comps}")
                global_step += 1
                continue

            # ---- backward + clip + step ----
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.grad_clip))
            if not math.isfinite(gnorm):
                # AMP overflow path: skip and let the scaler shrink loss_scale.
                skipped += 1
                scaler.update()
                global_step += 1
                continue
            scaler.step(optim)
            scaler.update()

            # ---- logging ----
            running["total"]  += comps["total"]
            running["sisnr"]  += comps["sisnr"]
            running["mrstft"] += comps["mrstft"]
            running["perc"]   += comps.get("perc", 0.0)
            running["gnorm"]  += gnorm
            running["n"]      += 1

            if (global_step + 1) % cfg.log_every == 0:
                n = running["n"]
                print(
                    f"ep {epoch:02d} step {global_step+1:6d} "
                    f"lr={lr:.2e} "
                    f"total={running['total']/n:+.4f} "
                    f"sisnr={running['sisnr']/n:+.4f} "
                    f"mr={running['mrstft']/n:.4f} "
                    f"perc={running['perc']/n:.4f} "
                    f"|g|={running['gnorm']/n:.3f} "
                    f"skipped={skipped}"
                )
                for k in ("total", "sisnr", "mrstft", "perc", "gnorm"):
                    running[k] = 0.0
                running["n"] = 0

            global_step += 1

        # ---- end-of-epoch: validate, save-best, plateau handling ----
        val_sisnr = validate_sisnr(model, val_loader, device)
        improved = val_sisnr > best_sisnr + cfg.min_delta_db
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s "
              f"val_sisnr={val_sisnr:+.3f} dB best={best_sisnr:+.3f} "
              f"(perceptual={loss_fn.perceptual})")

        if improved:
            best_sisnr = val_sisnr
            plateau = 0
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'epoch': epoch, 'step': global_step,
                'best': best_sisnr, 'cfg_tag': cfg_tag, 'cfg': cfg.__dict__,
            }, best_path)
            print(f'[ckpt] new best {best_sisnr:+.3f} dB -> {best_path}')
        else:
            plateau += 1
            print(f'[ckpt] no improvement (val={val_sisnr:+.3f} vs best '
                  f'{best_sisnr:+.3f}) plateau={plateau}/{cfg.patience}')
            if plateau >= cfg.patience:
                warm_restart(optim, cfg.warm_restart_lr_mult, cfg.warm_restart_lr_cap)
                plateau = 0  # give the restart room to recover
            if plateau >= cfg.hard_stop_patience:
                print(f'[train] no progress for {plateau} epochs -> stopping')
                break

        # Light "last" checkpoint for crash-resume (cheap, infrequent).
        if cfg.save_last_every and (epoch % cfg.save_last_every == 0):
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'epoch': epoch, 'step': global_step,
                'best': best_sisnr, 'cfg_tag': cfg_tag, 'cfg': cfg.__dict__,
            }, last_path)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-max", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--perceptual-start", type=int, default=8)
    p.add_argument("--finetune", action="store_true",
                   help="Fine-tune preset: lower LR, perceptual+TV losses ON.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Checkpoint path to resume/warm-start from.")
    p.add_argument("--resume-mode", type=str, default='warm',
                   choices=['full', 'warm', 'partial', 'none'],
                   help="full=resume opt+sched; warm=weights only; "
                        "partial=strict=False; none=ignore checkpoint.")
    p.add_argument("--ckpt-dir", type=str, default='.')
    p.add_argument("--min-resume-sisnr", type=float, default=5.0,
                   help="Below this dB, downgrade FULL resume to WARM.")
    p.add_argument("--patience", type=int, default=3)
    a = p.parse_args()
    cfg = TrainConfig(
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr_max=a.lr_max,
        warmup_steps=a.warmup_steps,
        workers=a.workers,
        amp=not a.no_amp,
        perceptual_start_epoch=a.perceptual_start,
        finetune=a.finetune,
        ckpt_dir=a.ckpt_dir,
        resume_from=a.resume_from,
        resume_mode=a.resume_mode,
        min_resume_sisnr_db=a.min_resume_sisnr,
        patience=a.patience,
    )
    if a.finetune:
        # Stage 2 (perceptual refinement): expects --resume-from <stage1 best>
        # Use 'warm' resume so optimizer moments from stage 1 don't anchor us.
        cfg.epochs = max(a.epochs, 10)
        cfg.lr_max = min(a.lr_max, 5e-5)        # 5e-5 -> 1e-6 cosine
        cfg.lr_start = 1e-6
        cfg.warmup_steps = 200
        cfg.w_sisnr = 0.3
        cfg.w_mrstft = 1.0
        cfg.w_mel = 0.8              # stronger mel weight in fine-tune
        cfg.w_complex = 1.5          # heavier phase weight in stage 2
        cfg.w_perc = 1.0
        cfg.w_mask_tv = 0.05
        cfg.si_snr_max_db = 25.0
        if cfg.resume_mode == 'full':
            print('[finetune] forcing resume_mode=warm for stage 2 stability')
            cfg.resume_mode = 'warm'
    return cfg


if __name__ == "__main__":
    train(parse_args())
