"""Loss functions for S-Gate training.

Includes:
  * Bounded SI-SNR loss (clamped to prevent gradient explosions when est ~ ref)
  * Multi-resolution STFT loss (spectral convergence + log-magnitude)
  * Combined SGateLoss that returns component dict for logging
  * Optional perceptual (speech-band) weighting

Design notes
------------
* SI-SNR is in dB and unbounded above (e.g. 60+ dB on near-perfect estimates),
  which makes its gradient blow up and dominate the total loss late in training.
  We clamp `-SI-SNR` from below at -SI_SNR_MAX (default 30 dB) so the spectral
  term remains meaningful and gradients stay well-behaved.
* MR-STFT is `spectral_convergence + log_magnitude` per resolution, averaged
  across resolutions (~O(1)).
* Default weights `w_sisnr=0.5, w_mrstft=1.0` keep both terms in the same
  numerical range. Typical early-training magnitudes: SI-SNR ~ 10-20,
  MR-STFT ~ 1-3.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frequency weighting curve (1-4 kHz emphasis)
# ---------------------------------------------------------------------------

def perceptual_freq_weights(
    n_freq: int,
    sample_rate: int = 16000,
    f_lo: float = 1000.0,
    f_hi: float = 4000.0,
    boost: float = 2.0,
    base: float = 1.0,
) -> torch.Tensor:
    """Smooth weighting that boosts the [f_lo, f_hi] band.

    Returns a 1-D tensor of length `n_freq`. Energy outside the band gets
    weight `base` (=1); inside the band, weight ramps up to `base + boost` (=3)
    via a raised-cosine, then ramps back down. This is what gives the biggest
    PESQ lift for speech in the 2-3 kHz formant region.
    """
    freqs = torch.linspace(0, sample_rate / 2, n_freq)
    w = torch.full_like(freqs, base)
    in_band = (freqs >= f_lo) & (freqs <= f_hi)
    if in_band.any():
        # Raised-cosine inside the band, peaks in the middle.
        x = (freqs[in_band] - f_lo) / (f_hi - f_lo)              # [0, 1]
        bump = 0.5 - 0.5 * torch.cos(2 * 3.141592653589793 * x)  # [0, 1, 0]
        w[in_band] = base + boost * bump
    return w


# ---------------------------------------------------------------------------
# SI-SNR (bounded)
# ---------------------------------------------------------------------------

def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SNR in dB. Inputs: [B, T]. Returns [B]."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target = dot * ref / ref_energy
    e_noise = est - s_target
    ratio = (s_target.pow(2).sum(dim=-1) + eps) / (e_noise.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio)


def si_snr_loss(est: torch.Tensor, ref: torch.Tensor, max_db: float = 30.0) -> torch.Tensor:
    """Negative SI-SNR clamped at -max_db so a single near-perfect example
    cannot dominate the batch / inflate gradients."""
    snr = si_snr(est, ref).clamp(max=max_db)
    return -snr.mean()


# ---------------------------------------------------------------------------
# Multi-resolution STFT loss
# ---------------------------------------------------------------------------

class STFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int,
                 freq_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        if freq_weight is None:
            freq_weight = torch.ones(n_fft // 2 + 1)
        # Reshape for broadcasting over [B, F, T]
        self.register_buffer("freq_weight", freq_weight.view(1, -1, 1), persistent=False)

    def _mag(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window, center=True, return_complex=True,
        )
        return spec.abs().clamp_min(1e-7)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        m_est, m_ref = self._mag(est), self._mag(ref)
        # Frequency-weighted spectral convergence + log-mag.
        w = self.freq_weight
        diff = (m_ref - m_est) * w
        sc  = torch.norm(diff, p="fro") / (torch.norm(m_ref * w, p="fro") + 1e-7)
        log_diff = (torch.log(m_est) - torch.log(m_ref)).abs() * w
        mag = log_diff.mean()
        return sc + mag


class MultiResSTFTLoss(nn.Module):
    """Mean of frequency-weighted STFT losses across multiple resolutions.

    Default resolutions aligned to model's n_fft=256 operating range.
    Using 256/512/1024 (not 512/1024/2048) avoids penalizing spectral
    detail the model cannot reconstruct with its 129-bin mask.
    """

    def __init__(
        self,
        n_ffts: Sequence[int] = (256, 512, 1024),
        hops:   Sequence[int] = (64, 128, 256),
        wins:   Sequence[int] = (256, 512, 1024),
        sample_rate: int = 16000,
        freq_emphasis: bool = True,
    ):
        super().__init__()
        assert len(n_ffts) == len(hops) == len(wins)
        losses = []
        for n, h, w in zip(n_ffts, hops, wins):
            fw = perceptual_freq_weights(n // 2 + 1, sample_rate=sample_rate) \
                if freq_emphasis else None
            losses.append(STFTLoss(n, h, w, freq_weight=fw))
        self.losses = nn.ModuleList(losses)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return sum(l(est, ref) for l in self.losses) / len(self.losses)


# ---------------------------------------------------------------------------
# Mel-spectrogram L1 loss (lightweight perceptual proxy)
# ---------------------------------------------------------------------------

class MelLoss(nn.Module):
    """L1 loss on log-mel spectrograms. Cheap perceptual proxy — mel scale
    approximates human frequency perception without a separate model."""

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 hop: int = 128, n_mels: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.register_buffer(
            'mel_fb',
            self._mel_filterbank(sample_rate, n_fft, n_mels),
            persistent=False,
        )
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)

    @staticmethod
    def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
        """Build a mel filterbank matrix [n_freq, n_mels]."""
        n_freq = n_fft // 2 + 1
        low_mel = 0.0
        high_mel = 2595.0 * math.log10(1.0 + (sr / 2) / 700.0)
        mels = torch.linspace(low_mel, high_mel, n_mels + 2)
        hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        bins = (hz / (sr / n_fft)).long().clamp(0, n_freq - 1)
        fb = torch.zeros(n_freq, n_mels)
        for m in range(n_mels):
            lo, mid, hi = int(bins[m]), int(bins[m + 1]), int(bins[m + 2])
            if mid > lo:
                fb[lo:mid, m] = torch.linspace(0, 1, mid - lo)
            if hi > mid:
                fb[mid:hi, m] = torch.linspace(1, 0, hi - mid)
        return fb

    def _log_mel(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=self.window,
                          center=True, return_complex=True)
        power = spec.abs().pow(2)                           # [B, F, T]
        mel = torch.matmul(power.transpose(-2, -1), self.mel_fb)  # [B, T, M]
        return torch.log10(mel.clamp_min(1e-7))

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return (self._log_mel(est) - self._log_mel(ref)).abs().mean()


# ---------------------------------------------------------------------------
# Complex spectrogram loss (PHASE-AWARE — critical for PESQ)
# ---------------------------------------------------------------------------

class ComplexSpecLoss(nn.Module):
    """L1 loss on real & imaginary parts of the STFT.

    Magnitude-only losses (SI-SNR, MR-STFT, mel) are PHASE-BLIND. Two signals
    with identical magnitude but different phase will have zero magnitude
    loss but sound completely different (PESQ collapses).

    By penalizing real+imag directly, we force the model's complex ratio mask
    to learn correct phase — the single biggest lever for PESQ improvement
    when going from magnitude-mask (PESQ ~1.7) to cRM (PESQ ~2.5+).

    Compression p<1 (Pyhann/Wisdom 2020) emphasizes low-energy bins where
    speech harmonics live, instead of being dominated by formant peaks.
    """

    def __init__(self, n_fft: int = 512, hop: int = 128, win: int = 512,
                 compress: float = 0.3):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.compress = compress
        self.register_buffer('window', torch.hann_window(win), persistent=False)

    def _spec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.win, window=self.window,
                          center=True, return_complex=True)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        s_est = self._spec(est)
        s_ref = self._spec(ref)
        # Power-law compression on magnitude, preserve phase direction
        mag_e = s_est.abs().clamp_min(1e-7).pow(self.compress)
        mag_r = s_ref.abs().clamp_min(1e-7).pow(self.compress)
        c_est = mag_e * torch.exp(1j * torch.angle(s_est))
        c_ref = mag_r * torch.exp(1j * torch.angle(s_ref))
        # All terms are .mean() across (B, F, T) -> automatically O(1)
        # regardless of segment length / batch size, so AMP fp16 is stable.
        l_mag = (mag_e - mag_r).abs().mean()
        l_re  = (c_est.real - c_ref.real).abs().mean()
        l_im  = (c_est.imag - c_ref.imag).abs().mean()
        # Average the three terms so the loss range stays comparable to mel/MR.
        return (l_mag + l_re + l_im) / 3.0


# ---------------------------------------------------------------------------
# Combined loss (returns components for logging)
# ---------------------------------------------------------------------------

class SGateLoss(nn.Module):
    """L = w_sisnr * clamp(-SI-SNR)
         + w_mrstft * MR-STFT
         + w_mel * Mel-L1
         + w_complex * ComplexSpec  (PHASE-AWARE — main PESQ driver)
         + (optional) w_perc * MR-STFT(speech_band)
         + w_mask_tv * TV(mask)

    Recommended weights:
        w_sisnr  = 0.3    # bounded time-domain target (reduced)
        w_mrstft = 0.7    # spectral magnitude
        w_mel    = 0.5    # perceptual mel proxy
        w_complex= 1.0    # phase-aware (largest weight — drives PESQ)
        w_perc   = 0.3    # speech-band emphasis (delayed start)
        w_mask_tv= 0.02   # temporal smoothness (reduces musical noise)
    """

    def __init__(
        self,
        w_sisnr: float = 0.3,
        w_mrstft: float = 0.7,
        w_mel: float = 0.5,
        w_complex: float = 1.0,
        w_perc: float = 0.3,
        w_mask_tv: float = 0.02,
        si_snr_max_db: float = 30.0,
        perceptual: bool = False,
        freq_emphasis: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.w_sisnr = w_sisnr
        self.w_mrstft = w_mrstft
        self.w_mel = w_mel
        self.w_complex = w_complex
        self.w_perc = w_perc
        self.w_mask_tv = w_mask_tv
        self.si_snr_max_db = si_snr_max_db
        self.perceptual = perceptual
        self.sample_rate = sample_rate
        self.mrstft = MultiResSTFTLoss(sample_rate=sample_rate,
                                       freq_emphasis=freq_emphasis)
        self.mel_loss = MelLoss(sample_rate=sample_rate)
        self.complex_loss = ComplexSpecLoss()

    def set_perceptual(self, on: bool) -> None:
        """Toggle perceptual term (used for the warm-start schedule)."""
        self.perceptual = on

    def set_stage(self, stage: str) -> None:
        """Switch loss curriculum.

        Stages:
          'warmup'   : magnitude only (no complex/perceptual).
                       Use for first 2-3 epochs to let model learn basic
                       gain estimation without phase chaos. Stable gradients.
          'standard' : add complex spec loss (phase-aware). Default training.
          'perceptual': add speech-band perceptual term. Use after PESQ plateaus.
        """
        if stage == 'warmup':
            self.w_complex = 0.0
            self.w_mel = 0.3        # gentle mel guidance
            self.perceptual = False
        elif stage == 'standard':
            self.w_complex = 1.0
            self.w_mel = 0.5
            self.perceptual = False
        elif stage == 'perceptual':
            self.w_complex = 1.0
            self.w_mel = 0.5
            self.perceptual = True
        else:
            raise ValueError(f'unknown stage: {stage}')
        print(f'[loss] stage={stage} '
              f'w_sisnr={self.w_sisnr} w_mr={self.w_mrstft} '
              f'w_mel={self.w_mel} w_cplx={self.w_complex} '
              f'perc={self.perceptual} w_tv={self.w_mask_tv}')

    def _speech_band(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop, win = 512, 128, 512
        window = torch.hann_window(win, device=x.device)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=window, center=True, return_complex=True)
        freqs = torch.linspace(0, self.sample_rate / 2, spec.shape[-2], device=x.device)
        mask = ((freqs >= 300.0) & (freqs <= 3400.0)).float().unsqueeze(0).unsqueeze(-1)
        spec = spec * mask
        return torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=win,
                           window=window, center=True, length=x.shape[-1])

    def forward(
        self, est: torch.Tensor, ref: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_sisnr = si_snr_loss(est, ref, max_db=self.si_snr_max_db)
        l_mr = self.mrstft(est, ref)
        l_mel = self.mel_loss(est, ref)
        l_cplx = self.complex_loss(est, ref)
        total = (self.w_sisnr * l_sisnr + self.w_mrstft * l_mr +
                 self.w_mel * l_mel + self.w_complex * l_cplx)
        comps = {"sisnr": float(l_sisnr.detach()),
                 "mrstft": float(l_mr.detach()),
                 "mel": float(l_mel.detach()),
                 "complex": float(l_cplx.detach())}
        if self.perceptual:
            l_perc = self.mrstft(self._speech_band(est), self._speech_band(ref))
            total = total + self.w_perc * l_perc
            comps["perc"] = float(l_perc.detach())
        if self.w_mask_tv > 0 and mask is not None:
            tv_t = (mask[..., 1:] - mask[..., :-1]).abs().mean()
            tv_f = (mask[:, 1:, :] - mask[:, :-1, :]).abs().mean()
            l_tv = tv_t + 0.5 * tv_f
            total = total + self.w_mask_tv * l_tv
            comps["mask_tv"] = float(l_tv.detach())
        comps["total"] = float(total.detach())
        return total, comps
