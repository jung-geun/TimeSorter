#!/usr/bin/env python3
"""Schema vs Content 통과율 차트 — 포맷 미준수 vs 추론 능력 분리.

content_n30.json(base/instruct) + SFT/DPO 알려진 수치로
'스키마 준수 통과' vs '내용만 통과' 두 막대를 모델별로 비교.
"""
from __future__ import annotations

import json
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
GRAY, BLUE, NAVY, ORANGE, RED, TEXT = "#9AA3AC", "#3D6FB4", "#27406B", "#E08A3C", "#C0392B", "#222222"

# n=150 4-way 결과 (eval_q35_4way_*_n150.json)
def _load(key):
    d = json.loads(Path(f"outputs/eval_q35_4way_{key}_n150.json").read_text())
    n = d["total"]
    return d["schema_pass"] / n * 100, d["content_pass"] / n * 100


_ins = _load("instruct_base"); _base = _load("base_no_rlhf")
_sft = _load("sft_q35"); _dpo = _load("dpo_q35")
# 모델별 (label, schema%, content%)
models = [
    ("Qwen3.5-4B\n(no FT)", *_ins),
    ("Base\n(no FT)",        *_base),
    ("+ SFT v4",             *_sft),
    ("+ DPO v5",             *_dpo),
]

fig, ax = plt.subplots(figsize=(10, 5.4))
x = np.arange(len(models))
w = 0.36
schema_v = [m[1] for m in models]
content_v = [m[2] for m in models]
ax.bar(x - w/2, schema_v, w, label="스키마 준수 통과 (포맷 요구)", color=GRAY, zorder=3)
ax.bar(x + w/2, content_v, w, label="내용만 통과 (포맷 무시, 추론만)", color=BLUE, zorder=3)
for xi, v in zip(x - w/2, schema_v):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=11.5, color=TEXT)
for xi, v in zip(x + w/2, content_v):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=BLUE)

ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in models], fontsize=12)
ax.set_ylim(0, 110)
ax.set_ylabel("통과율 (%)", fontsize=12)
ax.set_title("포맷(스키마) vs 추론(내용) 분리 통과율 (Qwen3.5-4B, n=150)",
             fontsize=15, fontweight="bold", pad=14)
ax.legend(fontsize=11.5, loc="upper left", frameon=False)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#CCC"); ax.spines["bottom"].set_color("#CCC")
ax.tick_params(length=0); ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#EAEAEA", linewidth=1)

# no-FT 갭 강조
ax.annotate("0% → 34%\n추론은 했지만\n포맷만 틀림", xy=(0 + w/2, 34.0), xytext=(0.55, 62),
            ha="center", fontsize=11, fontweight="bold", color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="#EAF1F9", ec=BLUE, alpha=0.9))

plt.tight_layout()
out = ASSETS / "chart_schema_vs_content.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[saved] {out}")
