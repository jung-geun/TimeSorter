#!/usr/bin/env python
"""학습 지표 4B vs 9B 비교 차트 (SFT 동일 데이터 / DPO 데이터 다름)."""
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
B4, B9, TEXT, GRAY = "#3D6FB4", "#5FA86E", "#222222", "#888"

# 지표 (4B: RTX 3080 Ti 12GB / 9B: RTX 4090 24GB)
SFT = {"loss": (0.309, 0.320), "acc": (90.8, 91.5), "hours": (9.5, 11.5)}   # 동일 데이터 sft_v4_train 6,056
DPO = {"acc": (97.5, 88.0), "margin": (3.50, 0.099), "loss": (0.166, 0.650)}  # 데이터 다름

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
fig.suptitle("학습 지표 비교 — Qwen3.5 4B vs 9B", fontsize=15, fontweight="bold")

# ── 좌: SFT (동일 데이터 — 공정 비교) ──
ax = axes[0]
groups = ["token\naccuracy (%)", "train_loss\n(×100)", "학습시간\n(h)"]
v4 = [SFT["acc"][0], SFT["loss"][0]*100, SFT["hours"][0]]
v9 = [SFT["acc"][1], SFT["loss"][1]*100, SFT["hours"][1]]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, v4, w, label="4B (RTX 3080 Ti)", color=B4, zorder=3)
ax.bar(x + w/2, v9, w, label="9B (RTX 4090)", color=B9, zorder=3)
fmt = ["{:.1f}", "{:.1f}", "{:.1f}h"]
for xi, val, f in zip(x - w/2, v4, fmt):
    ax.text(xi, val + 1.5, f.format(val), ha="center", fontsize=10, color=TEXT)
for xi, val, f in zip(x + w/2, v9, fmt):
    ax.text(xi, val + 1.5, f.format(val), ha="center", fontsize=10, color=TEXT)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
ax.set_ylim(0, 105); ax.set_ylabel("값", fontsize=11)
ax.set_title("SFT (동일 데이터 sft_v4_train 6,056 — 공정 비교)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", frameon=False)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.tick_params(length=0); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEE")
ax.text(0.5, -16, "→ 9B SFT 성능은 4B와 거의 동일 (acc +0.7%p, loss·시간 유사)",
        transform=ax.transData, fontsize=9, color=GRAY, ha="left")

# ── 우: DPO (데이터 다름 — 직접 비교 불가) ──
ax = axes[1]
groups = ["reward\naccuracy (%)", "reward\nmargin", "train_loss"]
v4 = [DPO["acc"][0], DPO["margin"][0], DPO["loss"][0]]
v9 = [DPO["acc"][1], DPO["margin"][1], DPO["loss"][1]]
# 스케일 차이 커서 정규화 대신 텍스트 라벨 + 막대(acc는 %, margin/loss는 작음)
x = np.arange(3); w = 0.36
# acc는 0-100, margin/loss는 작은 값 → 별도 처리 위해 acc만 막대, 나머지 텍스트
ax.bar(0 - w/2, v4[0], w, color=B4, zorder=3, label="4B (DPO v5 on-policy)")
ax.bar(0 + w/2, v9[0], w, color=B9, zorder=3, label="9B (DPO v4_extra)")
ax.text(0 - w/2, v4[0] + 1.5, f"{v4[0]:.1f}", ha="center", fontsize=10)
ax.text(0 + w/2, v9[0] + 1.5, f"{v9[0]:.1f}", ha="center", fontsize=10)
# margin·loss는 텍스트 표로
ax.text(1.5, 75, "reward margin", ha="center", fontsize=10.5, fontweight="bold", color=TEXT)
ax.text(1.5, 65, f"4B {v4[1]:.2f}  vs  9B {v9[1]:.3f}", ha="center", fontsize=11, color=TEXT)
ax.text(1.5, 48, "train_loss", ha="center", fontsize=10.5, fontweight="bold", color=TEXT)
ax.text(1.5, 38, f"4B {v4[2]:.3f}  vs  9B {v9[2]:.3f}", ha="center", fontsize=11, color=TEXT)
ax.set_xticks([0, 1.5]); ax.set_xticklabels(["reward\naccuracy (%)", "margin · loss"], fontsize=10.5)
ax.set_xlim(-0.6, 2.3); ax.set_ylim(0, 105); ax.set_ylabel("값", fontsize=11)
ax.set_title("DPO (데이터 다름 — 직접 비교 불가)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower left", frameon=False)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.tick_params(length=0); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEE")
ax.text(-0.55, -16, "→ 4B는 on-policy hard-neg(margin↑), 9B는 v4_extra — 차이는 데이터 탓",
        transform=ax.transData, fontsize=9, color=GRAY, ha="left")

plt.tight_layout(rect=[0, 0.03, 1, 1])
out = ASSETS / "chart_train_metrics_4b_9b.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[saved] {out}")
