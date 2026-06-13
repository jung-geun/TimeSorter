#!/usr/bin/env python
"""Qwen3.5 4B vs 9B 비교 차트 — 모델 크기별 시나리오 통과율.

데이터 출처:
  4B: experiments(n=30, DPO v5)  9B: experiments/4090_1x_9b_v4 (n=150, docker verify_chosen)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_KO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_KO)
    matplotlib.rcParams["font.family"] = [fm.FontProperties(fname=_KO).get_name(), "DejaVu Sans"]
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150

ASSETS = Path("assets")
BLUE, GREEN, RED, TEXT = "#3D6FB4", "#5FA86E", "#C0392B", "#222222"

SC = ["날짜 혼재", "당일 시각", "리스크", "상대 날짜", "의존성 체인"]
# 최종 모델 기준
M4B = {"overall": 90.0, "n": 30, "sc": [100, 100, 80, 100, 67]}    # Qwen3.5-4B DPO v5 (n=30)
M9B = {"overall": 88.7, "n": 150, "sc": [98, 97, 95, 93, 57]}      # Qwen3.5-9B DPO v4 (n=150)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
fig.suptitle("Qwen3.5 4B vs 9B — 시나리오별 통과율", fontsize=15, fontweight="bold")

# 좌: 전체 통과율
ax1 = axes[0]
bars = ax1.bar([0, 1], [M4B["overall"], M9B["overall"]], color=[BLUE, GREEN],
               width=0.55, zorder=3)
for x, m in zip([0, 1], [M4B, M9B]):
    ax1.text(x, m["overall"] + 1, f"{m['overall']:.1f}%", ha="center", va="bottom",
             fontsize=16, fontweight="bold", color=TEXT)
    ax1.text(x, -5, f"n={m['n']}", ha="center", fontsize=8, color="#666")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Qwen3.5-4B", "Qwen3.5-9B"], fontsize=12)
ax1.set_ylim(0, 105); ax1.set_ylabel("통과율 (%)", fontsize=12)
ax1.set_title("전체 통과율", fontsize=12, fontweight="bold")
ax1.axhline(90, color=RED, ls="--", lw=0.8, alpha=0.4)
for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)
ax1.tick_params(length=0); ax1.set_axisbelow(True); ax1.yaxis.grid(True, color="#EEE")

# 우: 시나리오별
ax2 = axes[1]
x = np.arange(len(SC)); w = 0.38
ax2.bar(x - w/2, M4B["sc"], w, label=f"4B (n={M4B['n']})", color=BLUE, zorder=3)
ax2.bar(x + w/2, M9B["sc"], w, label=f"9B (n={M9B['n']})", color=GREEN, zorder=3)
for xi, v in zip(x - w/2, M4B["sc"]):
    ax2.text(xi, v + 1.5, f"{v}", ha="center", fontsize=10, color=TEXT)
for xi, v in zip(x + w/2, M9B["sc"]):
    ax2.text(xi, v + 1.5, f"{v}", ha="center", fontsize=10, color=TEXT)
ax2.set_xticks(x); ax2.set_xticklabels(SC, fontsize=11)
ax2.set_ylim(0, 112); ax2.set_ylabel("통과율 (%)", fontsize=12)
ax2.set_title("시나리오별 통과율", fontsize=12, fontweight="bold")
ax2.legend(fontsize=11, loc="lower left", frameon=False)
for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
ax2.tick_params(length=0); ax2.set_axisbelow(True); ax2.yaxis.grid(True, color="#EEE")
ax2.annotate("9B도 체인은\n약점 (57%)", xy=(4 + w/2, 57), xytext=(3.4, 32),
             ha="center", fontsize=10.5, fontweight="bold", color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", fc="#FBECEA", ec=RED, alpha=0.9))

plt.tight_layout()
out = ASSETS / "chart_4b_vs_9b.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[saved] {out}")
print("주의: 4B는 n=30(빠른 평가), 9B는 n=150(전체 held-out) — 표본 크기 다름.")
