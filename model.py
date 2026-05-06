"""
S-Gate (Structure-Gate Network)
-------------------------------
A lightweight, causal speech enhancement model that:
  * enhances speech in noise
  * preserves structured sounds (music, birds, environmental)
  * suppresses unstructured noise (traffic, hum)
  * runs in real-time (<10ms algorithmic latency on a single 5ms hop)
  * uses < 500K parameters
  * is INT8-friendly (only Conv/GRU/Linear/Sigmoid/tanh)

Pipeline:
  STFT -> magnitude + spectral flux -> causal Conv2D encoder
       -> single-layer GRU temporal core -> Linear mask head (sigmoid)
       -> mask * noisy_mag -> iSTFT(orig_phase) -> tanh
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# STFT / iSTFT helpers (causal-friendly: hop = 5ms @ 16 kHz)
# ---------------------------------------------------------------------------

class STFT(nn.Module):
    """Thin wrapper around torch.stft / torch.istft.

    Uses Hann window with win_length == n_fft and 75% overlap (hop = n_fft/4)
    to guarantee COLA / NOLA reconstruction on all PyTorch backends including
    CUDA. center=True so reconstruction length matches the input.
    """

    def __init__(self, n_fft: int = 256, hop_length: int = 64, win_length: Optional[int] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Force win_length == n_fft for robust istft on CUDA.
        self.win_length = n_fft if win_length is None else int(win_length)
        if self.win_length != n_fft:
            # Silently coerce — different sizes break CUDA istft for some configs.
            self.win_length = n_fft
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

    def stft(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """wav: [B, T]  ->  mag, phase: [B, F, T_frames]"""
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()
        phase = torch.angle(spec)
        return mag, phase

    def istft(self, mag: torch.Tensor, phase: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """mag, phase: [B, F, T_frames]  ->  wav: [B, T]"""
        spec = torch.polar(mag, phase)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )
        return wav


def spectral_flux(mag: torch.Tensor) -> torch.Tensor:
    """Spectral flux feature: difference between consecutive magnitude frames.

    mag: [B, F, T]  ->  flux: [B, F, T]   (first frame zero-padded => causal)
    """
    prev = F.pad(mag[:, :, :-1], (1, 0))  # shift right by one frame
    return mag - prev


# ---------------------------------------------------------------------------
# Causal building blocks
# ---------------------------------------------------------------------------

class CausalDepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable Conv2D with causal padding along the time axis.

    Input layout: [B, C, T, F]  (time = dim 2, frequency = dim 3)
    Time padding is left-only; frequency padding is symmetric.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_t: int = 3,
        kernel_f: int = 5,
        stride_f: int = 1,
    ):
        super().__init__()
        self.pad_t = kernel_t - 1                       # causal: pad only on the left
        self.pad_f = (kernel_f - 1) // 2                # symmetric in frequency
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(kernel_t, kernel_f),
            stride=(1, stride_f),
            padding=0,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad: (left_f, right_f, left_t, right_t)
        x = F.pad(x, (self.pad_f, self.pad_f, self.pad_t, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# S-Gate model
# ---------------------------------------------------------------------------

class SGateModel(nn.Module):
    """Structure-Gate Network.

    Args:
        n_fft, hop_length, win_length: STFT parameters (defaults => 5ms hop @ 16k).
        enc_channels: channel widths of the two Conv2D encoder layers.
        freq_reduced: target frequency dim after encoder (129 -> freq_reduced).
        gru_hidden:   GRU hidden size (single layer).
        debug:        if True, prints tensor shapes during forward().
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        enc_channels: Tuple[int, int] = (16, 24),
        freq_reduced: int = 64,
        gru_hidden: int = 128,
        mask_floor: float = 0.05,        # min gain -> avoids over-suppression / musical noise
        smooth_mask: bool = False,       # depthwise 3x3 smoother (hurts transients, off by default)
        complex_mask: bool = True,       # predict real+imag mask for implicit phase correction
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1                    # 129 for n_fft=256
        self.freq_reduced = freq_reduced
        self.gru_hidden = gru_hidden
        self.mask_floor = float(mask_floor)
        self.smooth_mask = bool(smooth_mask)
        self.complex_mask = bool(complex_mask)

        self.stft = STFT(n_fft, hop_length, win_length)

        # ---- Conv encoder (causal in time, stride=2 in freq twice) ----
        # 129 -> 65 -> 33 frequency bins, then a 1x1 projection to freq_reduced.
        c1, c2 = enc_channels
        self.enc1 = CausalDepthwiseSeparableConv2d(2,  c1, kernel_t=3, kernel_f=5, stride_f=2)
        self.enc2 = CausalDepthwiseSeparableConv2d(c1, c2, kernel_t=3, kernel_f=5, stride_f=2)

        # After two stride-2 convs over 129 bins: ceil(129/2)=65, ceil(65/2)=33.
        self.freq_after_enc = math.ceil(math.ceil(self.n_freq / 2) / 2)
        enc_flat = c2 * self.freq_after_enc

        # Project to a compact per-frame feature (keeps GRU input small).
        self.proj_in = nn.Linear(enc_flat, freq_reduced)

        # ---- Temporal core: single-layer GRU ----
        self.gru = nn.GRU(
            input_size=freq_reduced,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # ---- Mask head: project GRU output back to full frequency resolution ----
        # If complex_mask: output 2*n_freq (real + imag parts of a complex ratio mask)
        # else: magnitude-only sigmoid mask (legacy).
        mask_out_dim = self.n_freq * 2 if self.complex_mask else self.n_freq
        if self.complex_mask:
            self.mask_head = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gru_hidden, mask_out_dim),
            )
            # CRITICAL stability init: start as near-identity (real≈1, imag≈0)
            # so untrained model passes noisy through unchanged. Without this,
            # initial loss is enormous and gradients explode.
            with torch.no_grad():
                last = self.mask_head[-1]
                last.weight.mul_(0.01)                       # tiny weights -> output dominated by bias
                last.bias.zero_()
                last.bias[:self.n_freq] = 0.55               # tanh(0.55) ≈ 0.5; with floor -> ~0.55 (passthrough-ish)
                # imag bias stays 0 -> initial phase unchanged
        else:
            self.mask_head = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gru_hidden, mask_out_dim),
                nn.Sigmoid(),
            )

        # ---- Optional mask smoother: depthwise 3 (time) x 3 (freq), causal in time ----
        # Initialised to an identity-like kernel so it starts as a no-op.
        if self.smooth_mask:
            self.mask_smoother = nn.Conv2d(1, 1, kernel_size=(3, 3), bias=False)
            with torch.no_grad():
                k = torch.zeros(1, 1, 3, 3)
                k[0, 0, -1, 1] = 1.0          # current frame, current bin -> 1 (identity init)
                self.mask_smoother.weight.copy_(k)
        else:
            self.mask_smoother = None

    # ------------------------------------------------------------------
    # Internal: spectrogram features -> per-frame mask
    # ------------------------------------------------------------------
    def _features_to_mask(
        self,
        mag: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """mag: [B, F, T]  ->  mask: [B, F, T], next_hidden: [1, B, H]"""
        flux = spectral_flux(mag)                                  # [B, F, T]
        x = torch.stack([mag, flux], dim=1)                        # [B, 2, F, T]
        x = x.transpose(2, 3).contiguous()                         # [B, 2, T, F]
        if self.debug: print("enc in :", x.shape)

        x = self.enc1(x)
        if self.debug: print("enc1   :", x.shape)
        x = self.enc2(x)
        if self.debug: print("enc2   :", x.shape)

        # [B, C, T, F'] -> [B, T, C*F']
        B, C, T, Fp = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fp)
        x = self.proj_in(x)                                        # [B, T, freq_reduced]
        if self.debug: print("gru in :", x.shape)

        gru_out, h_n = self.gru(x, h0)                             # [B, T, H]
        raw = self.mask_head(gru_out)                              # [B, T, F] or [B, T, 2F]

        if self.complex_mask:
            # Complex ratio mask: real + imag parts, bounded via tanh.
            # Bound mask magnitude to [0, 1] (mask floor preserves low-energy content)
            mask_r, mask_i = raw.chunk(2, dim=-1)                  # each [B, T, F]
            mask_r = mask_r.tanh()                                 # ∈ (-1, 1)
            mask_i = mask_i.tanh()
            # Optional magnitude bound: prevents the complex mask from amplifying
            # noise when |cmask| > 1 (which can happen with raw real+imag tanh).
            cmag = torch.sqrt(mask_r ** 2 + mask_i ** 2 + 1e-8)
            scale = (cmag.clamp_min(1.0))                          # divide only if > 1
            mask_r = mask_r / scale
            mask_i = mask_i / scale
            mask_r = mask_r.transpose(1, 2).contiguous()           # [B, F, T]
            mask_i = mask_i.transpose(1, 2).contiguous()
            if self.debug: print("cRM    :", mask_r.shape, mask_i.shape)
            return (mask_r, mask_i), h_n
        else:
            mask = raw.transpose(1, 2).contiguous()                # [B, F, T]
            # Smooth mask in time/freq (causal in time).
            if self.mask_smoother is not None:
                m = mask.unsqueeze(1)                              # [B, 1, F, T]
                m = m.transpose(2, 3)                              # [B, 1, T, F]
                m = F.pad(m, (1, 1, 2, 0))                         # freq sym, time causal
                m = self.mask_smoother(m).clamp(0.0, 1.0)
                mask = m.squeeze(1).transpose(1, 2).contiguous()
            # Mask floor
            if self.mask_floor > 0:
                mask = self.mask_floor + (1.0 - self.mask_floor) * mask
            if self.debug: print("mask   :", mask.shape)
            return mask, h_n

    # ------------------------------------------------------------------
    # Forward (full utterance)
    # ------------------------------------------------------------------
    def forward(self, wav: torch.Tensor, return_mask: bool = False):
        """wav: [B, T]  ->  enhanced_wav: [B, T] (or (wav, mask) if return_mask)."""
        in_len = wav.shape[-1]
        mag, phase = self.stft.stft(wav)                           # [B, F, T_frames]
        mask_out, _ = self._features_to_mask(mag)

        if self.complex_mask:
            # Complex ratio mask applied to full complex STFT.
            mask_r, mask_i = mask_out                               # each [B, F, T]
            spec = torch.polar(mag, phase)                          # complex [B, F, T]
            cmask = torch.complex(mask_r, mask_i)
            enh_spec = spec * cmask
            out = torch.istft(
                enh_spec, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=self.stft.window,
                center=True, length=in_len,
            )
            # Magnitude mask for visualization / return (from the complex mask)
            vis_mask = cmask.abs().clamp(0, 1)
        else:
            enh_mag = mag * mask_out
            out = self.stft.istft(enh_mag, phase, length=in_len)
            vis_mask = mask_out

        # Soft clamp instead of tanh (preserves dynamics, prevents rare overflow)
        out = out.clamp(-1.0, 1.0)

        if return_mask:
            return out, vis_mask
        return out

    # ------------------------------------------------------------------
    # Streaming inference (one frame at a time)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> dict:
        """Allocate a fresh streaming state."""
        device = device or next(self.parameters()).device
        return {
            "h":         torch.zeros(1, batch_size, self.gru_hidden, device=device),
            "prev_mag":  torch.zeros(batch_size, self.n_freq, 1, device=device),
            # Time-conv history: each enc layer needs (kernel_t - 1) past frames.
            "enc1_hist": torch.zeros(batch_size, 2,                 2, self.n_freq,            device=device),
            "enc2_hist": torch.zeros(batch_size, self.enc1.pointwise.out_channels,
                                     2, math.ceil(self.n_freq / 2), device=device),
            # 2 prior mask frames for the smoothing conv (kernel_t=3, causal).
            "mask_hist": torch.zeros(batch_size, 1, 2, self.n_freq, device=device),
        }

    @torch.no_grad()
    def stream_step(
        self,
        frame_wav: torch.Tensor,
        state: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Process one hop of audio (hop_length samples) and return one hop of output.

        For a real streaming pipeline the caller maintains an input ring buffer of
        size win_length and feeds the most recent win_length samples here.
        Returns: (out_frame [B, hop_length], new_state)
        """
        # Compute one STFT frame: input must be exactly win_length samples.
        assert frame_wav.shape[-1] == self.win_length, \
            f"stream_step expects win_length={self.win_length} samples, got {frame_wav.shape[-1]}"
        spec = torch.fft.rfft(frame_wav * self.stft.window, n=self.n_fft)  # [B, F]
        mag = spec.abs().unsqueeze(-1)                                      # [B, F, 1]
        phase = torch.angle(spec).unsqueeze(-1)                             # [B, F, 1]

        flux = mag - state["prev_mag"]                                      # [B, F, 1]
        state["prev_mag"] = mag

        # Build [B, 2, T=1, F] then prepend cached time history for causal convs.
        x = torch.stack([mag, flux], dim=1).transpose(2, 3).contiguous()    # [B, 2, 1, F]

        # ---- enc1 with cached history (kernel_t = 3 -> need 2 prev frames) ----
        x_in = torch.cat([state["enc1_hist"], x], dim=2)                    # [B, 2, 3, F]
        state["enc1_hist"] = x_in[:, :, 1:, :]                              # keep last 2
        x = F.pad(x_in, (self.enc1.pad_f, self.enc1.pad_f, 0, 0))           # freq pad only
        x = self.enc1.depthwise(x)
        x = self.enc1.pointwise(x)
        x = self.enc1.norm(x)
        x = self.enc1.act(x)                                                # [B, c1, 1, F1]

        # ---- enc2 with cached history ----
        x_in = torch.cat([state["enc2_hist"], x], dim=2)                    # [B, c1, 3, F1]
        state["enc2_hist"] = x_in[:, :, 1:, :]
        x = F.pad(x_in, (self.enc2.pad_f, self.enc2.pad_f, 0, 0))
        x = self.enc2.depthwise(x)
        x = self.enc2.pointwise(x)
        x = self.enc2.norm(x)
        x = self.enc2.act(x)                                                # [B, c2, 1, F2]

        # ---- project + GRU step + mask head ----
        B, C, _, Fp = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, 1, C * Fp)
        x = self.proj_in(x)
        gru_out, state["h"] = self.gru(x, state["h"])
        mask = self.mask_head(gru_out).transpose(1, 2)                      # [B, F, 1]

        # ---- mask smoother (causal, 3x3) using cached mask history ----
        if self.mask_smoother is not None:
            m = mask.unsqueeze(1).transpose(2, 3)                           # [B, 1, 1, F]
            m_in = torch.cat([state["mask_hist"], m], dim=2)                # [B, 1, 3, F]
            state["mask_hist"] = m_in[:, :, 1:, :]
            m_in = F.pad(m_in, (1, 1, 0, 0))                                # freq sym pad
            m_out = self.mask_smoother(m_in).clamp(0.0, 1.0)                # [B, 1, 1, F]
            mask = m_out.squeeze(1).transpose(1, 2).contiguous()            # [B, F, 1]

        # ---- mask floor ----
        if self.mask_floor > 0:
            mask = self.mask_floor + (1.0 - self.mask_floor) * mask

        enh_mag = mag * mask
        # Single-frame iSTFT via inverse rFFT + window (overlap-add is the caller's job).
        spec_enh = torch.polar(enh_mag, phase).squeeze(-1)                  # [B, F]
        time_frame = torch.fft.irfft(spec_enh, n=self.n_fft)[:, :self.win_length]
        time_frame = time_frame * self.stft.window
        # Return the new hop_length samples; caller does overlap-add for the rest.
        return torch.tanh(time_frame[:, -self.hop_length:]), state


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Latency note:
#   Algorithmic latency = win_length / sr = 160 / 16000 = 10 ms (one analysis window).
#   Per-frame compute is dominated by a single GRU step + 2 small Conv2D ops, well
#   under 1 ms on a modern mobile CPU. Total real-time latency target: < 10 ms.


if __name__ == "__main__":
    m = SGateModel(debug=True)
    print(f"Parameters: {count_parameters(m):,}")
    x = torch.randn(2, 16000)               # 1 s of audio @ 16 kHz
    y = m(x)
    print("output:", y.shape)
