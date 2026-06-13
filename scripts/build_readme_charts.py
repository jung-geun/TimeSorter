#!/usr/bin/env python3
"""README용 차트 생성 — assets/ 디렉토리에 저장 (git 추적 가능).

생성:
  assets/chart_learning_curve.png   전체 학습 커브 (DPO v3 → Qwen3.5 DPO v5)
  assets/chart_qwen35_comparison.png  Qwen3.5 4종 비교 (Base/Instruct/SFT/DPO)
  assets/chart_scenario_heatmap.png   시나리오별 통과율 히트맵
  assets/chart_violation_trend.png    위반 유형 감소 추이
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import matplotlib.font_manager as fm

# Noto Sans CJK KR 등록 (한글 라벨 렌더링)
_KO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_KO_FONT)
    _ko_name = fm.FontProperties(fname=_KO_FONT).get_name()
    matplotlib.rcParams["font.family"] = [_ko_name, "DejaVu Sans", "sans-serif"]
except Exception:
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

# ── 데이터 ───────────────────────────────────────────────────────────────────
# Qwen3-4B 체크포인트 (150샘플 eval)
QWEN3_CPS = [
    ("DPO v3\n(Qwen3-4B)",  "outputs/eval_dpo_v3.json"),
    ("GRPO v4\n(Qwen3-4B)", "outputs/eval_grpo_v4.json"),
    ("SFT v4\n(Qwen3-4B)",  "outputs/eval_sft_v4.json"),
    ("DPO v5\n(Qwen3-4B)",  "outputs/eval_dpo_v5.json"),
]
# Qwen3.5-4B 체크포인트 (30샘플 eval)
QWEN35_CPS = [
    ("Qwen3.5-4B\n(no FT)",  "outputs/eval_qwen35_instruct_base_n30.json", "#DD8452", ""),
    ("Qwen3.5-4B-Base\n(no FT)", "outputs/eval_qwen35_base_no_rlhf_n30.json", "#55A868", ""),
    ("SFT v4\n(Qwen3.5-4B)", "outputs/eval_qwen35_sft_q35_n30.json", "#DD8452", "///"),
    ("DPO v5\n(Qwen3.5-4B)", None, "#DD8452", "xxx"),   # manual
]
DPO_V5_Q35_MANUAL = {
    "clean": 27, "total": 30,
    "per_scenario": {
        "dated_mixed":      {"n": 9,  "violated": 0},
        "intraday":         {"n": 6,  "violated": 0},
        "risk":             {"n": 5,  "violated": 1},
        "relative":         {"n": 4,  "violated": 0},
        "dependency_chain": {"n": 6,  "violated": 2},
    },
    "per_kind": {"chain": 4, "past_rank": 1},
}

SCENARIOS    = ["dated_mixed", "intraday", "risk", "relative", "dependency_chain"]
SC_LABELS    = ["Dated\nMixed", "Intraday", "Risk", "Relative", "Dep. Chain"]

def _load(path):
    return json.loads(Path(path).read_text())

def _pct(d): return d["clean"] / d["total"] * 100

def _per_sc(d):
    return {sc: (d["per_scenario"][sc]["n"] - d["per_scenario"][sc].get("violated", 0))
                / d["per_scenario"][sc]["n"] * 100
            for sc in d["per_scenario"]}


# ── Chart 1: 학습 커브 (Learning Curve) ─────────────────────────────────────
def chart_learning_curve():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Qwen3-4B 점들
    q3_labels, q3_vals, q3_n = [], [], []
    for label, fpath in QWEN3_CPS:
        d = _load(fpath)
        q3_labels.append(label); q3_vals.append(_pct(d)); q3_n.append(d["total"])

    # Qwen3.5-4B 점들 (연속)
    q35_vals = [
        _pct(_load("outputs/eval_qwen35_instruct_base_n30.json")),
        _pct(_load("outputs/eval_qwen35_base_no_rlhf_n30.json")),
        _pct(_load("outputs/eval_qwen35_sft_q35_n30.json")),
        90.0,
    ]
    q35_labels = ["Qwen3.5-4B\n(no FT)", "Qwen3.5-Base\n(no FT)",
                  "SFT v4\n(Qwen3.5)", "DPO v5\n(Qwen3.5)"]
    q35_n      = [30, 30, 30, 30]

    all_labels = q3_labels + q35_labels
    x = np.arange(len(all_labels))
    n_q3 = len(q3_labels)

    # Qwen3-4B 막대 (파란색)
    for i in range(n_q3):
        stage = "DPO" if "DPO" in q3_labels[i] else ("GRPO" if "GRPO" in q3_labels[i] else "SFT")
        hatch = "///" if stage == "SFT" else ("xxx" if stage == "DPO" else "...")
        ax.bar(i, q3_vals[i], color="#4C72B0", hatch=hatch, alpha=0.85,
               edgecolor="black", linewidth=0.8, width=0.65)
        ax.text(i, q3_vals[i] + 0.8, f"{q3_vals[i]:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Qwen3.5-4B 막대 (오렌지/초록)
    q35_colors = ["#DD8452", "#55A868", "#DD8452", "#DD8452"]
    q35_hatches = ["", "", "///", "xxx"]
    for j, (lbl, val, color, hatch) in enumerate(zip(q35_labels, q35_vals, q35_colors, q35_hatches)):
        i = n_q3 + j
        ax.bar(i, val, color=color, hatch=hatch, alpha=0.85,
               edgecolor="black", linewidth=0.8, width=0.65)
        ax.text(i, val + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(i, -5.5, f"n={q35_n[j]}", ha="center", fontsize=7, color="#666")

    for i, n in enumerate(q3_n):
        ax.text(i, -5.5, f"n={n}", ha="center", fontsize=7, color="#666")

    # 구분선
    ax.axvline(n_q3 - 0.5, color="#333", ls="--", lw=1.2, alpha=0.5)
    ax.text(n_q3 - 0.5 + 0.05, 102, "↑ Model Upgrade", fontsize=8,
            color="#333", va="top")

    # 주요 마일스톤 주석
    ax.annotate("+20.6%p\nData curation\n+ Loss masking",
                xy=(2, 77.3), xytext=(1.2, 88),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.1),
                fontsize=7.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="#ccc", alpha=0.9))
    ax.annotate("+12.7%p\nQwen3.5 upgrade",
                xy=(n_q3 + 2, 90.0), xytext=(n_q3 + 1.2, 100),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.1),
                fontsize=7.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.25", fc="#ffe0b2", ec="#ccc", alpha=0.9))

    ax.axhline(90, color="red",    ls="--", lw=0.9, alpha=0.45, label="90% target")
    ax.axhline(77.3, color="gray", ls=":",  lw=0.9, alpha=0.55, label="SFT v4 baseline (77.3%)")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_title("TimeSorter — Training Progression (Pass Rate)", fontsize=12, fontweight="bold")
    ax.set_ylim(-12, 112)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    legend_patches = [
        mpatches.Patch(facecolor="#4C72B0",  label="Qwen3-4B-Instruct"),
        mpatches.Patch(facecolor="#DD8452",  label="Qwen3.5-4B"),
        mpatches.Patch(facecolor="#55A868",  label="Qwen3.5-4B-Base"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="black", label="SFT"),
        mpatches.Patch(facecolor="white", hatch="xxx", edgecolor="black", label="DPO"),
        mpatches.Patch(facecolor="white", hatch="...", edgecolor="black", label="GRPO"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left", ncol=3,
              framealpha=0.85)

    plt.tight_layout()
    out = ASSETS / "chart_learning_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# ── Chart 2: Qwen3.5 4종 비교 (2-panel) ─────────────────────────────────────
def chart_qwen35_comparison():
    entries = [
        ("Qwen3.5-4B\n(no adapter)", _load("outputs/eval_qwen35_instruct_base_n30.json"),
         "#DD8452", "", "BASE"),
        ("Qwen3.5-4B-Base\n(no adapter)", _load("outputs/eval_qwen35_base_no_rlhf_n30.json"),
         "#55A868", "", "BASE"),
        ("SFT v4\n(Qwen3.5-4B)", _load("outputs/eval_qwen35_sft_q35_n30.json"),
         "#DD8452", "///", "SFT"),
        ("DPO v5\n(Qwen3.5-4B)", DPO_V5_Q35_MANUAL,
         "#DD8452", "xxx", "DPO"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Qwen3.5-4B — Base / Instruct / SFT / DPO Comparison",
                 fontsize=12, fontweight="bold")

    # Left: overall pass rate
    ax1 = axes[0]
    x = np.arange(len(entries))
    for i, (label, d, color, hatch, stage) in enumerate(entries):
        pct = _pct(d)
        ax1.bar(i, pct, color=color, hatch=hatch, alpha=0.87,
                edgecolor="black", linewidth=0.8, width=0.6)
        ax1.text(i, pct + 0.8, f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax1.text(i, -5, f"n={d['total']}", ha="center", fontsize=7.5, color="#555")

    ax1.set_xticks(x)
    ax1.set_xticklabels([e[0] for e in entries], fontsize=8.5)
    ax1.set_ylim(-10, 110)
    ax1.set_ylabel("Pass Rate (%)", fontsize=10)
    ax1.set_title("Overall Task Completion Rate", fontsize=10, fontweight="bold")
    ax1.axhline(90, color="red", ls="--", lw=0.9, alpha=0.5, label="90%")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.legend(fontsize=8)

    # Right: per-scenario grouped bar
    ax2 = axes[1]
    n_m = len(entries); n_sc = len(SCENARIOS)
    width = 0.75 / n_m
    xsc = np.arange(n_sc)
    for j, (label, d, color, hatch, stage) in enumerate(entries):
        sc_rates = _per_sc(d)
        vals = [sc_rates.get(sc, 0.0) for sc in SCENARIOS]
        offsets = xsc + (j - n_m / 2 + 0.5) * width
        ax2.bar(offsets, vals, width=width * 0.92, color=color, hatch=hatch,
                alpha=0.87, edgecolor="black", linewidth=0.5,
                label=label.replace("\n", " "))

    ax2.set_xticks(xsc)
    ax2.set_xticklabels(SC_LABELS, fontsize=9)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Pass Rate (%)", fontsize=10)
    ax2.set_title("Per-Scenario Task Handling Capability", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right", ncol=1)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")

    # 기여도 화살표
    for j_base, j_sft in [(0, 2)]:
        for k, sc in enumerate(SCENARIOS):
            base_v = _per_sc(entries[j_base][1]).get(sc, 0)
            sft_v  = _per_sc(entries[j_sft][1]).get(sc, 0)
            if sft_v - base_v > 30:
                xpos = xsc[k] + (j_sft - n_m / 2 + 0.5) * width
                ax2.annotate("", xy=(xpos, sft_v + 2), xytext=(xpos, base_v + 2),
                             arrowprops=dict(arrowstyle="->", color="#333", lw=1))

    legend_patches = [
        mpatches.Patch(facecolor="#DD8452", label="Qwen3.5-4B (Instruct)"),
        mpatches.Patch(facecolor="#55A868", label="Qwen3.5-4B-Base"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="black", label="+ SFT"),
        mpatches.Patch(facecolor="white", hatch="xxx", edgecolor="black", label="+ DPO"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    out = ASSETS / "chart_qwen35_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# ── Chart 3: 시나리오별 히트맵 (전체 체크포인트) ────────────────────────────
def chart_scenario_heatmap():
    all_rows = []
    for label, fpath in QWEN3_CPS:
        d = _load(fpath)
        all_rows.append((label.replace("\n", " "), _per_sc(d)))
    q35_files = [
        ("Qwen3.5-4B (no FT)",  "outputs/eval_qwen35_instruct_base_n30.json"),
        ("Qwen3.5-Base (no FT)","outputs/eval_qwen35_base_no_rlhf_n30.json"),
        ("SFT v4 (Qwen3.5)",    "outputs/eval_qwen35_sft_q35_n30.json"),
    ]
    for label, fpath in q35_files:
        all_rows.append((label, _per_sc(_load(fpath))))
    all_rows.append(("DPO v5 (Qwen3.5)", _per_sc(DPO_V5_Q35_MANUAL)))

    labels = [r[0] for r in all_rows]
    matrix = np.array([[r[1].get(sc, 0) for sc in SCENARIOS] for r in all_rows])

    fig, ax = plt.subplots(figsize=(9, len(labels) * 0.55 + 1.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(SC_LABELS, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_title("Per-Scenario Pass Rate Heatmap (%)", fontsize=11, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(SCENARIOS)):
            val = matrix[i, j]
            color = "white" if val < 45 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    # 구분선 Qwen3-4B vs Qwen3.5
    ax.axhline(len(QWEN3_CPS) - 0.5, color="white", lw=2)

    plt.colorbar(im, ax=ax, label="Pass Rate (%)", shrink=0.8)
    plt.tight_layout()
    out = ASSETS / "chart_scenario_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# ── Chart 4: 위반 유형 감소 추이 ────────────────────────────────────────────
def chart_violation_trend():
    rows = []
    for label, fpath in QWEN3_CPS:
        d = _load(fpath)
        rows.append((label.replace("\n", " "), d.get("per_kind", {})))

    q35_q35sft  = _load("outputs/eval_qwen35_sft_q35_n30.json")
    rows.append(("SFT v4 (Qwen3.5)", q35_q35sft.get("per_kind", {})))
    rows.append(("DPO v5 (Qwen3.5)", DPO_V5_Q35_MANUAL["per_kind"]))

    labels    = [r[0] for r in rows]
    viol_defs = [
        ("past_rank",       "#e6194b", "Past Rank"),
        ("chain",           "#3cb44b", "Chain Order"),
        ("intraday_order",  "#f58231", "Intraday Order"),
        ("none_first",      "#911eb4", "None First"),
        ("parse_or_count",  "#a9a9a9", "Parse Error"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for key, color, vlabel in viol_defs:
        vals = np.array([r[1].get(key, 0) for r in rows], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=vlabel,
               alpha=0.85, edgecolor="white", linewidth=0.4, width=0.6)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 2:
                ax.text(i, b + v / 2, str(int(v)), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals

    # 총 위반 수 표시
    totals = [sum(r[1].get(k, 0) for k, *_ in viol_defs) for r in rows]
    for i, t in enumerate(totals):
        ax.text(i, bottom[i] + 0.5, f"Total\n{int(t)}", ha="center", va="bottom",
                fontsize=7.5, color="#333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Violation Count", fontsize=10)
    ax.set_title("Violation Type Breakdown by Checkpoint", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # 구분선
    ax.axvline(len(QWEN3_CPS) - 0.5, color="#333", ls="--", lw=1.1, alpha=0.5)
    ax.text(len(QWEN3_CPS) - 0.5 + 0.05, ax.get_ylim()[1] * 0.97,
            "Qwen3.5 →", fontsize=8, color="#333", va="top")

    plt.tight_layout()
    out = ASSETS / "chart_violation_trend.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    chart_learning_curve()
    chart_qwen35_comparison()
    chart_scenario_heatmap()
    chart_violation_trend()
    print(f"\nAll charts → {ASSETS}/")
