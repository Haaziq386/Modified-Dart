#!/usr/bin/env python3
"""
Create a spectral complementarity panel for paper figures.

The figure shows:
1) Raw time-domain signal
2) FFT magnitude spectrum with dominant peaks highlighted

Example:
python scripts/paper_figures/plot_spectral_complementarity.py \
  --input datasets/ETT-small/ETTh1.csv \
  --column OT \
  --sample-start 0 \
  --length 336 \
  --output outputs/figures/spectral_complementarity.png
"""

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _load_signal(path: str, column: str, sample_start: int, length: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        if column is None:
            # Skip likely timestamp column if present.
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError("No numeric columns found in CSV.")
            col = numeric_cols[0]
        else:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in {path}.")
            col = column
        signal = df[col].to_numpy(dtype=np.float64)

    elif ext == ".npy":
        arr = np.load(path)
        # Accept 1D, 2D, or 3D and reduce to a 1D trace.
        if arr.ndim == 1:
            signal = arr.astype(np.float64)
        elif arr.ndim == 2:
            # Use first channel as default.
            signal = arr[:, 0].astype(np.float64)
        elif arr.ndim == 3:
            # Common shape [N, L, C] -> first sample/channel.
            signal = arr[0, :, 0].astype(np.float64)
        else:
            raise ValueError(f"Unsupported npy shape {arr.shape}.")

    else:
        raise ValueError("Unsupported file type. Use .csv or .npy")

    end = sample_start + length
    if sample_start < 0 or end > len(signal):
        raise ValueError(
            f"Requested slice [{sample_start}:{end}] is outside signal length {len(signal)}."
        )

    return signal[sample_start:end]


def _fft(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    window = np.hanning(n)
    sig = signal * window
    spec = np.fft.rfft(sig)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, mag


def _top_peaks(freqs: np.ndarray, mag: np.ndarray, k: int = 3) -> np.ndarray:
    if len(mag) <= 2:
        return np.array([], dtype=int)

    # Ignore DC when searching for periodic components.
    peaks, _ = find_peaks(mag[1:])
    peaks = peaks + 1

    if len(peaks) == 0:
        # Fallback: pick largest non-DC bins.
        candidates = np.argsort(mag[1:])[::-1][:k] + 1
        return np.array(candidates, dtype=int)

    ranked = peaks[np.argsort(mag[peaks])[::-1]]
    return ranked[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot time-domain vs FFT complementarity.")
    parser.add_argument("--input", type=str, required=True, help="Path to .csv or .npy signal source")
    parser.add_argument("--column", type=str, default=None, help="CSV column name. Default: first numeric column")
    parser.add_argument("--sample-start", type=int, default=0, help="Start index for the plotted segment")
    parser.add_argument("--length", type=int, default=336, help="Number of time steps to visualize")
    parser.add_argument("--sampling-rate", type=float, default=1.0, help="Samples per unit time")
    parser.add_argument("--normalize", action="store_true", help="Z-normalize the selected segment")
    parser.add_argument("--peak-k", type=int, default=3, help="How many dominant spectral peaks to highlight")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")

    args = parser.parse_args()

    sig = _load_signal(args.input, args.column, args.sample_start, args.length)
    if args.normalize:
        std = np.std(sig)
        sig = (sig - np.mean(sig)) / (std + 1e-8)

    freqs, mag = _fft(sig, sampling_rate=args.sampling_rate)
    peak_idx = _top_peaks(freqs, mag, k=args.peak_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    # Left: waveform
    t = np.arange(len(sig))
    axes[0].plot(t, sig, color="#005f73", linewidth=1.6)
    axes[0].set_title("Time Domain Signal")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    # Right: magnitude spectrum
    axes[1].plot(freqs, mag, color="#bb3e03", linewidth=1.6)
    if len(peak_idx) > 0:
        axes[1].scatter(freqs[peak_idx], mag[peak_idx], color="#9b2226", s=34, zorder=3)
        for i in peak_idx:
            f = freqs[i]
            # Convert to period in sample units when possible.
            period = (args.sampling_rate / f) if f > 0 else np.inf
            axes[1].annotate(
                f"f={f:.3f}\\nP~{period:.1f}",
                (f, mag[i]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=8,
            )

    axes[1].set_title("FFT Magnitude Spectrum")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        "Spectral Complementarity: periodic structure becomes explicit in frequency domain",
        fontsize=11,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
