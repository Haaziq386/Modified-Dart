#!/usr/bin/env python3
"""
Plot t-SNE embeddings for HTulTS representations:
- time-only embedding
- freq-only embedding
- fused embedding
Colored by downstream per-sample error quartile.

This script runs checkpointed inference on a dataset split and computes:
- embedding vectors from internal HTulTS branches
- per-sample forecasting error from model predictions

Example:
python scripts/paper_figures/plot_tsne_embeddings.py \
  --checkpoint outputs/checkpoints/<setting>/ckpt_best.pth \
  --data ETTh1 \
  --root-path ./datasets/ETT-small \
  --data-path ETTh1.csv \
  --input-len 336 \
  --pred-len 96 \
  --enc-in 7 \
  --features M \
  --output outputs/figures/tsne_embeddings.png
"""

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_factory import data_provider
from models import HtulTS


def _make_args(cli: argparse.Namespace) -> SimpleNamespace:
    # Minimal args object compatible with data_provider and HtulTS.Model.
    return SimpleNamespace(
        task_name="finetune",
        downstream_task="forecast",
        data=cli.data,
        root_path=cli.root_path,
        data_path=cli.data_path,
        features=cli.features,
        target=cli.target,
        freq=cli.freq,
        embed=cli.embed,
        seasonal_patterns=cli.seasonal_patterns,
        input_len=cli.input_len,
        seq_len=cli.input_len,
        label_len=0,
        pred_len=cli.pred_len,
        enc_in=cli.enc_in,
        d_model=cli.d_model,
        num_workers=cli.num_workers,
        batch_size=cli.batch_size,
        use_norm=cli.use_norm,
        use_noise=0,
        noise_level=0.1,
        use_forgetting=cli.use_forgetting,
        forgetting_type=cli.forgetting_type,
        forgetting_rate=cli.forgetting_rate,
        tfc_weight=cli.tfc_weight,
        tfc_warmup_steps=0,
        use_real_imag=cli.use_real_imag,
        use_warping=cli.use_warping,
        projection_dim=cli.projection_dim,
        patch_len=cli.patch_len,
        stride=cli.stride,
        use_cpc=cli.use_cpc,
        cpc_lambda=0.1,
        cpc_freq_mask_ratio=0.2,
        cpc_time_mask_ratio=0.2,
        cpc_use_learned_mask=1,
        cpc_loss_type="l2",
        cpc_pos_emb_dim=64,
        cpc_hidden_dim=256,
    )


def _load_model(args: SimpleNamespace, checkpoint: str, device: torch.device) -> HtulTS.Model:
    model = HtulTS.Model(args).float().to(device)

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    missing = len(model_state) - len(filtered)
    if missing > 0:
        print(f"Warning: loaded partial checkpoint. matched={len(filtered)} missing_or_mismatch={missing}")

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def _normalize_like_forecast(wrapper: HtulTS.Model, x: torch.Tensor) -> torch.Tensor:
    if not wrapper.use_norm:
        return x
    means = torch.mean(x, dim=1, keepdim=True).detach()
    x_center = x - means
    stdevs = torch.sqrt(torch.var(x_center, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    return x_center / stdevs


def _extract_branch_embeddings(wrapper: HtulTS.Model, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reproduce the internal dual-branch representation path from LightweightModel.forward.
    lw = wrapper.model

    bsz, seq_len, num_feat = x_norm.size()
    x_flat = x_norm.permute(0, 2, 1).reshape(bsz * num_feat, seq_len)

    x_patches = lw.patch_embed(x_flat)
    x_patches_flat = x_patches.view(x_patches.size(0), -1)
    h_time = lw.patch_mixer(x_patches_flat)

    h_freq = lw.freq_backbone(x_flat, return_bins=False)
    h_fused = lw.fusion(h_time, h_freq)

    if lw.use_forgetting:
        h_fused = lw.forgetting_layer(h_fused, is_finetune=True)

    # Aggregate channel-wise vectors back to sample-wise vectors for t-SNE.
    h_time_s = h_time.view(bsz, num_feat, -1).mean(dim=1)
    h_freq_s = h_freq.view(bsz, num_feat, -1).mean(dim=1)
    h_fused_s = h_fused.view(bsz, num_feat, -1).mean(dim=1)

    return h_time_s, h_freq_s, h_fused_s


def _error_quartile_labels(errors: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    q1, q2, q3 = np.quantile(errors, [0.25, 0.5, 0.75])
    labels = np.digitize(errors, bins=[q1, q2, q3], right=True)  # 0..3

    names = [
        f"Q1 (<= {q1:.4f})",
        f"Q2 ({q1:.4f} - {q2:.4f})",
        f"Q3 ({q2:.4f} - {q3:.4f})",
        f"Q4 (> {q3:.4f})",
    ]
    return labels.astype(np.int64), names


def _fit_tsne(x: np.ndarray, seed: int) -> np.ndarray:
    n = x.shape[0]
    if n < 5:
        raise RuntimeError("Need at least 5 samples for t-SNE.")

    perplexity = min(30, max(5, n // 10))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot t-SNE for time/freq/fused HTulTS embeddings.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Dataset key, e.g. ETTh1")
    parser.add_argument("--root-path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--data-path", type=str, required=True, help="Data filename")
    parser.add_argument("--features", type=str, default="M", help="Forecasting feature mode: M/S/MS")
    parser.add_argument("--target", type=str, default="OT", help="Target column for S/MS")
    parser.add_argument("--freq", type=str, default="h", help="Data frequency code")
    parser.add_argument("--embed", type=str, default="timeF", help="Embedding type for loader")
    parser.add_argument("--seasonal-patterns", type=str, default="Monthly", help="Seasonal pattern for m4")

    parser.add_argument("--input-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--enc-in", type=int, required=True, help="Number of input channels")
    parser.add_argument("--d-model", type=int, default=512)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1000, help="Cap number of test samples for speed")

    parser.add_argument("--use-norm", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--use-real-imag", type=int, default=0)
    parser.add_argument("--use-warping", type=int, default=0)
    parser.add_argument("--use-forgetting", type=int, default=0)
    parser.add_argument("--forgetting-type", type=str, default="activation")
    parser.add_argument("--forgetting-rate", type=float, default=0.1)
    parser.add_argument("--tfc-weight", type=float, default=0.05)
    parser.add_argument("--use-cpc", type=int, default=0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    cli = parser.parse_args()

    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    if cli.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = _make_args(cli)
    model = _load_model(args, cli.checkpoint, device=device)

    _, test_loader = data_provider(args, flag="test")

    time_vecs: List[np.ndarray] = []
    freq_vecs: List[np.ndarray] = []
    fused_vecs: List[np.ndarray] = []
    sample_errors: List[np.ndarray] = []

    taken = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            x_norm = _normalize_like_forecast(model, batch_x)
            h_time, h_freq, h_fused = _extract_branch_embeddings(model, x_norm)

            preds = model.forecast(batch_x)
            f_dim = -1 if args.features == "MS" else 0
            y_true = batch_y[:, -args.pred_len :, f_dim:]
            y_pred = preds[:, -args.pred_len :, f_dim:]
            mse = ((y_pred - y_true) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()

            time_vecs.append(h_time.detach().cpu().numpy())
            freq_vecs.append(h_freq.detach().cpu().numpy())
            fused_vecs.append(h_fused.detach().cpu().numpy())
            sample_errors.append(mse)

            taken += batch_x.size(0)
            if taken >= cli.max_samples:
                break

    h_time_np = np.concatenate(time_vecs, axis=0)[: cli.max_samples]
    h_freq_np = np.concatenate(freq_vecs, axis=0)[: cli.max_samples]
    h_fused_np = np.concatenate(fused_vecs, axis=0)[: cli.max_samples]
    err_np = np.concatenate(sample_errors, axis=0)[: cli.max_samples]

    q_labels, q_names = _error_quartile_labels(err_np)

    emb_time = _fit_tsne(h_time_np, seed=cli.seed)
    emb_freq = _fit_tsne(h_freq_np, seed=cli.seed)
    emb_fused = _fit_tsne(h_fused_np, seed=cli.seed)

    cmap = ["#0a9396", "#94d2bd", "#ee9b00", "#bb3e03"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), constrained_layout=True)
    panels = [
        (axes[0], emb_time, "Time-only representation"),
        (axes[1], emb_freq, "Freq-only representation"),
        (axes[2], emb_fused, "Fused representation"),
    ]

    for ax, emb, title in panels:
        for q in range(4):
            m = q_labels == q
            if not np.any(m):
                continue
            ax.scatter(
                emb[m, 0],
                emb[m, 1],
                s=12,
                alpha=0.78,
                c=cmap[q],
                label=q_names[q],
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.2)

    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="best", fontsize=8)

    fig.suptitle(
        "t-SNE of Time/Frequency/Fused Embeddings colored by sample error quartile",
        fontsize=11,
    )

    out_dir = os.path.dirname(cli.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(cli.output, dpi=cli.dpi, bbox_inches="tight")
    print(f"Saved figure to: {cli.output}")


if __name__ == "__main__":
    main()
