#!/usr/bin/env python3
"""
Plot training dynamics for pretraining:
- L_recon (estimated as train_total - L_TFC)
- L_TFC

This parser targets pretrain logs that contain lines like:
Epoch: 1/10, Time: 12.34, Train Loss: 0.1234, TF-C Loss: 0.0042, Vali Loss: 0.1111

Example:
python scripts/paper_figures/plot_training_dynamics.py \
  --log-file outputs/logs/pretrain_ETTh1_20260102_092000.log \
  --output outputs/figures/training_dynamics.png
"""

import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+)\/(?:\d+).*?Train Loss:\s*(?P<train>[0-9eE+\-.]+).*?TF-C Loss:\s*(?P<tfc>[0-9eE+\-.]+)",
    re.IGNORECASE,
)


def parse_log(log_file: str) -> Dict[str, List[float]]:
    epochs: List[int] = []
    train_loss: List[float] = []
    tfc_loss: List[float] = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group("epoch")))
            train_loss.append(float(m.group("train")))
            tfc_loss.append(float(m.group("tfc")))

    if not epochs:
        raise RuntimeError(
            "No epoch lines with Train Loss and TF-C Loss found. "
            "Check that the selected log file is a pretrain log."
        )

    train_arr = np.array(train_loss, dtype=np.float64)
    tfc_arr = np.array(tfc_loss, dtype=np.float64)
    recon_arr = np.maximum(train_arr - tfc_arr, 0.0)

    return {
        "epoch": epochs,
        "train": train_arr.tolist(),
        "tfc": tfc_arr.tolist(),
        "recon": recon_arr.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot L_recon and L_TFC over epochs.")
    parser.add_argument("--log-file", type=str, required=True, help="Path to pretrain .log file")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--title", type=str, default="Training Dynamics", help="Figure title")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()

    data = parse_log(args.log_file)

    epochs = np.array(data["epoch"])
    l_recon = np.array(data["recon"])
    l_tfc = np.array(data["tfc"])

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax2 = ax1.twinx()

    l1 = ax1.plot(epochs, l_recon, color="#005f73", marker="o", linewidth=1.8, markersize=3.8, label="L_recon")[0]
    l2 = ax2.plot(epochs, l_tfc, color="#bb3e03", marker="s", linewidth=1.8, markersize=3.6, label="L_TFC")[0]

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L_recon", color="#005f73")
    ax2.set_ylabel("L_TFC", color="#bb3e03")
    ax1.tick_params(axis="y", labelcolor="#005f73")
    ax2.tick_params(axis="y", labelcolor="#bb3e03")
    ax1.grid(alpha=0.25)

    ax1.set_title(args.title)
    ax1.legend([l1, l2], ["L_recon", "L_TFC"], loc="upper right")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
