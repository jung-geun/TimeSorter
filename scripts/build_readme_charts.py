#!/usr/bin/env python3
"""README/발표용 차트 — 가독성 우선, 필요한 지표만.

생성 (assets/):
  chart_progression.png   Qwen3.5-4B 파인튜닝 단계별 통과율 (헤드라인, 4막대)
  chart_scenario.png      시나리오별 통과율 SFT v4 vs DPO v5 (체인 약점 강조)
  chart_milestones.png    학습 여정 마일스톤 (데이터 큐레이션·모델 업그레이드 점프)

설계 원칙: 솔리드 색상(해치 없음), 막대 최소화, 큰 라벨, 한 차트=한 메시지.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트
_KO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_KO_FONT)
    _ko = fm.FontProperties(fname=_KO_FONT).get_name()
    matplotlib.rcParams["font.family"] = [_ko, "DejaVu Sans"]
except Exception:
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

# 색상
GRAY   = "#9AA3AC"
GREEN  = "#5FA86E"
BLUE   = "#3D6FB4"
NAVY   = "#27406B"
ORANGE = "#E08A3C"
RED    = "#C0392B"
TEXT   = "#222222"

SCENARIOS = ["dated_mixed", "intraday", "risk", "relative", "dependency_chain"]
SC_KO = ["날짜 혼재", "당일 시각", "리스크", "상대 날짜", "의존성 체인"]


def _load(p):
    return json.loads(Path(p).read_text())


def _pct(d):
    return d["clean"] / d["total"] * 100


def _scrates(d):
    return {sc: (v["n"] - v.get("violated", 0)) / v["n"] * 100
            for sc, v in d["per_scenario"].items()}


DPO_Q35 = {
    "clean": 27, "total": 30,
    "per_scenario": {
        "dated_mixed": {"n": 9, "violated": 0}, "intraday": {"n": 6, "violated": 0},
        "risk": {"n": 5, "violated": 1}, "relative": {"n": 4, "violated": 0},
        "dependency_chain": {"n": 6, "violated": 2},
    },
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EAEAEA", linewidth=1)


# ── 1. 파인튜닝 단계별 통과율 (헤드라인) ─────────────────────────────────────
def chart_progression():
    labels = ["어댑터 없음\n(Instruct)", "Base\n(RLHF 미적용)", "+ SFT v4", "+ DPO v5"]
    vals   = [0.0, 16.7, 90.0, 90.0]
    colors = [GRAY, GREEN, BLUE, NAVY]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(4)
    bars = ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=TEXT)

    ax.axhline(90, color=RED, ls="--", lw=1, alpha=0.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel("통과율 (%)", fontsize=12)
    ax.set_title("Qwen3.5-4B 파인튜닝 단계별 통과율 (held-out n=30)",
                 fontsize=15, fontweight="bold", pad=14)
    _style(ax)

    # 핵심 화살표 주석
    ax.annotate("", xy=(2, 90), xytext=(0, 12),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2, alpha=0.7))
    ax.text(1.0, 58, "파인튜닝으로\n0% → 90%", ha="center", fontsize=12.5,
            fontweight="bold", color=BLUE,
            bbox=dict(boxstyle="round,pad=0.35", fc="#EAF1F9", ec=BLUE, alpha=0.9))

    plt.tight_layout()
    out = ASSETS / "chart_progression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


# ── 2. 시나리오별 통과율 (SFT vs DPO) ────────────────────────────────────────
def chart_scenario():
    sft = _scrates(_load("outputs/eval_qwen35_sft_q35_n30.json"))
    dpo = _scrates(DPO_Q35)
    sft_v = [sft.get(s, 0) for s in SCENARIOS]
    dpo_v = [dpo.get(s, 0) for s in SCENARIOS]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(SCENARIOS))
    w = 0.36
    ax.bar(x - w/2, sft_v, w, label="SFT v4", color=BLUE, zorder=3)
    ax.bar(x + w/2, dpo_v, w, label="DPO v5", color=ORANGE, zorder=3)
    for xi, v in zip(x - w/2, sft_v):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=11, color=TEXT)
    for xi, v in zip(x + w/2, dpo_v):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=11, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(SC_KO, fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_ylabel("통과율 (%)", fontsize=12)
    ax.set_title("시나리오별 통과율 — SFT v4 vs DPO v5 (Qwen3.5-4B)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(fontsize=12, loc="lower left", frameon=False)
    _style(ax)

    # 체인 약점 강조
    ax.annotate("공통 약점\n(능력 부재)", xy=(4, 67), xytext=(3.5, 40),
                ha="center", fontsize=11.5, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FBECEA", ec=RED, alpha=0.9))

    plt.tight_layout()
    out = ASSETS / "chart_scenario.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


# ── 3. 학습 여정 마일스톤 ────────────────────────────────────────────────────
def chart_milestones():
    labels = ["DPO v3\n(Qwen3-4B)", "SFT v4\n(Qwen3-4B)", "SFT+DPO\n(Qwen3.5-4B)"]
    vals   = [56.7, 77.3, 90.0]
    colors = [GRAY, BLUE, NAVY]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(3)
    ax.bar(x, vals, color=colors, width=0.55, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.8, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_ylabel("통과율 (%)", fontsize=12)
    ax.set_title("학습 여정 — 두 번의 핵심 점프", fontsize=15, fontweight="bold", pad=14)
    _style(ax)

    # 점프 주석
    ax.annotate("", xy=(1, 77.3), xytext=(0, 56.7),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.text(0.5, 70, "+20.6%p\n데이터 큐레이션\n+ Loss 마스킹", ha="center",
            fontsize=11, fontweight="bold", color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="#EAF1F9", ec=BLUE, alpha=0.9))
    ax.annotate("", xy=(2, 90.0), xytext=(1, 77.3),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    ax.text(1.5, 86, "+12.7%p\n모델 업그레이드", ha="center",
            fontsize=11, fontweight="bold", color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDF1E6", ec=ORANGE, alpha=0.9))

    plt.tight_layout()
    out = ASSETS / "chart_milestones.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    # 구버전 혼잡 차트 제거
    for old in ["chart_learning_curve.png", "chart_qwen35_comparison.png",
                "chart_scenario_heatmap.png", "chart_violation_trend.png"]:
        p = ASSETS / old
        if p.exists():
            p.unlink(); print(f"[removed] {p}")
    chart_progression()
    chart_scenario()
    chart_milestones()
    print(f"\nAll charts → {ASSETS}/")
