#!/usr/bin/env python3
"""TimeSorter Qwen3.5 벤치마크 시각화.

outputs/eval_qwen35_*.json 파일을 읽어 두 개의 차트를 생성:
  1. 전체 통과율 막대 (Base → SFT → DPO 진행)
  2. 시나리오별 통과율 히트맵

사용:
  uv run python scripts/plot_benchmark.py
  uv run python scripts/plot_benchmark.py --limit 30 --out outputs/benchmark_qwen35.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "NanumGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 표시 순서 및 스타일 ────────────────────────────────────────────────────
ORDER = [
    ("instruct_base",  "Qwen3.5-4B\n(no adapter)",      "#4C72B0", ""),
    ("base_no_rlhf",   "Qwen3.5-4B-Base\n(no adapter)", "#55A868", ""),
    ("sft_q35",        "SFT v4\n(Qwen3.5-4B)",          "#4C72B0", "///"),
    ("dpo_q35",        "DPO v5\n(Qwen3.5-4B)",          "#4C72B0", "xxx"),
    ("sft_q35_base",   "SFT v4\n(Qwen3.5-4B-Base)",     "#55A868", "///"),
    ("dpo_q35_base",   "DPO v5\n(Qwen3.5-4B-Base)",     "#55A868", "xxx"),
]

# DPO v5 Qwen3.5-4B는 이미 측정된 값 존재 (30샘플)
MANUAL_OVERRIDE = {
    "dpo_q35": {
        "label": "DPO v5\n(Qwen3.5-4B)",
        "total": 30, "clean": 27,
        "per_scenario": {
            "dated_mixed":      {"n": 9,  "violated": 0},
            "dependency_chain": {"n": 6,  "violated": 2},
            "intraday":         {"n": 6,  "violated": 0},
            "relative":         {"n": 4,  "violated": 0},
            "risk":             {"n": 5,  "violated": 1},
        },
        "per_kind": {"chain": 4, "past_rank": 1},
    },
}

SCENARIOS = ["dated_mixed", "intraday", "risk", "relative", "dependency_chain"]
SC_LABELS  = ["Dated\nMixed", "Intraday", "Risk", "Relative", "Dep.\nChain"]


def load_results(out_dir: str, limit: int) -> dict[str, dict]:
    """eval_qwen35_<key>_n<limit>.json 로드 + manual override 적용."""
    results: dict[str, dict] = {}

    for key, label, color, hatch in ORDER:
        path = Path(out_dir) / f"eval_qwen35_{key}_n{limit}.json"
        if path.exists():
            d = json.loads(path.read_text())
            results[key] = d
        elif key in MANUAL_OVERRIDE:
            results[key] = MANUAL_OVERRIDE[key]
        else:
            results[key] = None  # 미완료

    return results


def _pass_rate(d: dict) -> float:
    return d["clean"] / d["total"] * 100


def _per_sc_rates(d: dict) -> dict[str, float]:
    out = {}
    for sc, v in d["per_scenario"].items():
        n = v["n"]; viol = v.get("violated", 0)
        out[sc] = (n - viol) / n * 100 if n else 0.0
    return out


def make_chart(results: dict[str, dict | None], out: str) -> None:
    # 사용 가능한 항목만
    available = [(key, label, color, hatch)
                 for key, label, color, hatch in ORDER
                 if results.get(key) is not None]

    if not available:
        print("[warn] 결과 없음. benchmark_qwen35.py 먼저 실행하세요.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "TimeSorter — Qwen3.5-4B / Qwen3.5-4B-Base Benchmark",
        fontsize=14, fontweight="bold",
    )

    # ── Chart 1: 전체 통과율 ─────────────────────────────────────────────
    ax1 = axes[0]
    x = np.arange(len(available))
    for i, (key, label, color, hatch) in enumerate(available):
        d = results[key]
        pct = _pass_rate(d)
        ax1.bar(i, pct, color=color, hatch=hatch, alpha=0.85,
                edgecolor="black", linewidth=0.8)
        ax1.text(i, pct + 0.8, f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
        n = d.get("total", "?")
        ax1.text(i, -4, f"n={n}", ha="center", va="top", fontsize=7, color="gray")

    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label, _, _ in available], fontsize=8)
    ax1.set_ylabel("Pass Rate (%)")
    ax1.set_title("Overall Pass Rate")
    ax1.set_ylim(0, 110)
    ax1.axhline(y=90, color="red",    linestyle="--", alpha=0.5, linewidth=1, label="90%")
    ax1.grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(facecolor="#4C72B0", label="Qwen3.5-4B (Instruct)"),
        mpatches.Patch(facecolor="#55A868", label="Qwen3.5-4B-Base"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="black", label="SFT"),
        mpatches.Patch(facecolor="white", hatch="xxx", edgecolor="black", label="DPO"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc="upper left")

    # ── Chart 2: 시나리오별 통과율 ───────────────────────────────────────
    ax2 = axes[1]
    n_models = len(available)
    n_sc     = len(SCENARIOS)
    width    = 0.8 / max(n_models, 1)
    x_sc     = np.arange(n_sc)

    for j, (key, label, color, hatch) in enumerate(available):
        d = results[key]
        sc_rates = _per_sc_rates(d)
        vals = [sc_rates.get(sc, 0.0) for sc in SCENARIOS]
        offsets = x_sc + (j - n_models / 2 + 0.5) * width
        ax2.bar(offsets, vals, width=width * 0.9,
                color=color, hatch=hatch, alpha=0.85,
                edgecolor="black", linewidth=0.5,
                label=label.replace("\n", " "))

    ax2.set_xticks(x_sc)
    ax2.set_xticklabels(SC_LABELS, fontsize=9)
    ax2.set_ylabel("Pass Rate (%)")
    ax2.set_title("Per-Scenario Pass Rate")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=7, loc="upper left", ncol=2)

    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()

    # 요약 테이블 출력
    print("\n=== 결과 요약 ===")
    print(f"{'모델':<35} {'통과율':>8}  {'dated':>6} {'intra':>6} {'risk':>6} {'rel':>6} {'chain':>6}")
    print("-" * 75)
    for key, label, _, _ in available:
        d = results[key]
        pct = _pass_rate(d)
        sc  = _per_sc_rates(d)
        print(f"{label.replace(chr(10), ' '):<35} {pct:>7.1f}%"
              f"  {sc.get('dated_mixed', 0):>5.0f}%"
              f" {sc.get('intraday', 0):>5.0f}%"
              f" {sc.get('risk', 0):>5.0f}%"
              f" {sc.get('relative', 0):>5.0f}%"
              f" {sc.get('dependency_chain', 0):>5.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="outputs/benchmark_qwen35.png")
    parser.add_argument("--limit",  type=int, default=30)
    parser.add_argument("--dir",    default="outputs")
    args = parser.parse_args()

    results = load_results(args.dir, args.limit)
    make_chart(results, args.out)


if __name__ == "__main__":
    main()
