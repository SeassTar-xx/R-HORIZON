#!/usr/bin/env python3
"""
从 train_metrics_steps_all.jsonl 绘制训练曲线（单实验、单条曲线/子图）。
用法:
  python training/scripts/visualize_train_metrics.py \\
    --jsonl training/checkpoints/<run>/stats/train_metrics_steps_all.jsonl \\
    -o training/checkpoints/<run>/stats/train_curves.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def load_jsonl(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    steps: List[int] = []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = obj.get("global_steps")
            if s is None:
                s = obj.get("step")
            if s is None:
                continue
            steps.append(int(s))
            rows.append(obj)

    if not steps:
        raise SystemExit(f"未从 {path} 解析到任何有效行（需要 global_steps 或 step）")

    keys = set()
    for r in rows:
        keys.update(r.keys())

    def series(k: str) -> np.ndarray:
        out = np.full(len(rows), np.nan, dtype=float)
        for i, r in enumerate(rows):
            if k not in r:
                continue
            v = r[k]
            if v is None:
                continue
            if isinstance(v, (int, float)):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                out[i] = float(v)
        return out

    arrays: Dict[str, np.ndarray] = {}
    for k in keys:
        arrays[k] = series(k)

    return np.array(steps, dtype=int), arrays


def _plot_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    window: int,
    *,
    show_legend: bool = False,
) -> None:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    y_plot = _smooth(y, window) if window > 1 else y
    (line,) = ax.plot(x, y_plot, color=color, linewidth=1.4, label=label, alpha=0.95)
    ax.grid(True, linestyle="-", alpha=0.35, color="#bbbbbb")
    ax.set_xlabel("Training Steps", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    if show_legend:
        ax.legend(handles=[line], loc="best", fontsize=9, framealpha=0.9)


def main() -> None:
    p = argparse.ArgumentParser(description="可视化 train_metrics_steps_all.jsonl")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="指标 jsonl 路径（默认: 与 -o 同目录的 train_metrics_steps_all.jsonl）",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="输出 PNG 路径")
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        metavar="W",
        help="滑动平均窗口（步数），1 表示不平滑，与参考图类似的毛刺可设 1",
    )
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    out_path: Path = args.output
    jsonl_path = args.jsonl
    if jsonl_path is None:
        jsonl_path = out_path.parent / "train_metrics_steps_all.jsonl"
    if not jsonl_path.is_file():
        raise SystemExit(f"找不到 jsonl: {jsonl_path}")

    steps, arrs = load_jsonl(jsonl_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # 参考 TensorBoard：白底、浅灰网格
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#dddddd",
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = gridspec.GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.15, 1.0, 1.0],
        hspace=0.38,
        wspace=0.32,
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.07,
    )

    color_total = "#1f77b4"
    dim_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    dyn_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 行 0：总体奖励
    ax_total = fig.add_subplot(gs[0, :])
    if "total_reward" in arrs:
        _plot_line(
            ax_total,
            steps,
            arrs["total_reward"],
            color_total,
            "Total reward",
            args.smooth,
            show_legend=True,
        )
    ax_total.set_title("Reward (total)", fontsize=12, fontweight="medium")
    ax_total.set_ylabel("Reward", fontsize=10)

    # 行 1：各维度奖励（DenseChain）
    dim_specs = [
        ("prog_reward", "prog_reward", "Progress"),
        ("comp_reward", "comp_reward", "Completion"),
        ("neg_reward", "neg_reward", "Negative"),
        ("cons_reward", "cons_reward", "Consistency"),
        ("form_reward", "form_reward", "Format"),
    ]
    for i, (key, _, title) in enumerate(dim_specs):
        ax = fig.add_subplot(gs[1, i])
        if key in arrs:
            _plot_line(ax, steps, arrs[key], dim_colors[i], title, args.smooth, show_legend=False)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.set_ylabel("Reward", fontsize=9)

    # 第 6 列：放 critic/rewards/mean 若存在且与 total 不同视角（可选）；否则留空说明
    ax_extra = fig.add_subplot(gs[1, 5])
    alt = "critic/rewards/mean"
    if alt in arrs and np.any(np.isfinite(arrs[alt])):
        _plot_line(
            ax_extra,
            steps,
            arrs[alt],
            dim_colors[4],
            "critic rewards mean",
            args.smooth,
            show_legend=True,
        )
        ax_extra.set_title("Critic reward (mean)", fontsize=10, fontweight="medium")
        ax_extra.set_ylabel("Value", fontsize=9)
    else:
        # 只有 5 个维度时，第 6 格显示 dense LLM 成功率（若存在）
        if "dense_llm_success" in arrs and np.any(np.isfinite(arrs["dense_llm_success"])):
            calls = arrs.get("dense_llm_calls")
            succ = arrs["dense_llm_success"]
            if calls is not None and np.any(np.isfinite(calls)) and np.max(calls[np.isfinite(calls)]) > 0:
                ratio = np.divide(
                    succ,
                    np.maximum(calls, 1.0),
                    out=np.full_like(succ, np.nan),
                    where=np.isfinite(calls) & (calls > 0),
                )
                _plot_line(
                    ax_extra,
                    steps,
                    ratio,
                    dim_colors[4],
                    "LLM judge OK / calls",
                    args.smooth,
                    show_legend=True,
                )
                ax_extra.set_title("Dense LLM success ratio", fontsize=10, fontweight="medium")
                ax_extra.set_ylabel("Ratio", fontsize=9)
                ax_extra.set_ylim(-0.05, 1.05)
            else:
                ax_extra.axis("off")
        else:
            ax_extra.axis("off")

    # 修正：dim_specs 有 5 项，我用了 gs[1,0..4] 和 gs[1,5] 给 extra。前 5 个 ax 应对应 0..4
    # 上面循环是 enumerate(dim_specs) 对 i in 0..4，ax = gs[1,i] —— 对
    # ax_extra 是 gs[1,5] —— 对

    # 行 2：响应长度、每步耗时、熵损失（对齐参考图三栏）
    ax_rlen = fig.add_subplot(gs[2, 0:2])
    rlen_key = "response_length/mean" if "response_length/mean" in arrs else "response_length"
    if rlen_key in arrs:
        _plot_line(
            ax_rlen,
            steps,
            arrs[rlen_key],
            dyn_colors[0],
            "Response length",
            args.smooth,
            show_legend=True,
        )
    ax_rlen.set_title("Response Length", fontsize=11, fontweight="medium")
    ax_rlen.set_ylabel("Tokens / length", fontsize=10)

    ax_time = fig.add_subplot(gs[2, 2:4])
    tkey = "time_per_step" if "time_per_step" in arrs else "timing_s/step"
    if tkey in arrs:
        _plot_line(
            ax_time,
            steps,
            arrs[tkey],
            dyn_colors[1],
            "Time per step",
            args.smooth,
            show_legend=True,
        )
    ax_time.set_title("Time Per Step", fontsize=11, fontweight="medium")
    ax_time.set_ylabel("Time (s/step)", fontsize=10)

    ax_ent = fig.add_subplot(gs[2, 4:6])
    eloss = "entropy_loss" if "entropy_loss" in arrs else "actor/entropy_loss"
    if eloss in arrs:
        _plot_line(
            ax_ent,
            steps,
            arrs[eloss],
            dyn_colors[2],
            "Entropy loss",
            args.smooth,
            show_legend=True,
        )
    ax_ent.set_title("Entropy Loss", fontsize=11, fontweight="medium")
    ax_ent.set_ylabel("Entropy loss", fontsize=10)

    run_name = jsonl_path.parent.name
    fig.suptitle(f"Training curves — {run_name} (single run)", fontsize=12, y=1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
