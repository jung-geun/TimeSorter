#!/usr/bin/env python
"""9B 4-way 통계·차트 — base/no-sft/SFT/DPO (n=30) schema vs content.

outputs/eval_9b_*_n30.json 4개를 읽어:
  assets/chart_9b_schema_vs_content.png  포맷 vs 추론 (4B와 동일 구도)
  data/v6_audit/.. (X)
  + 콘솔 통계 표
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
GRAY, GREEN, BLUE, NAVY, RED, TEXT = "#9AA3AC", "#5FA86E", "#3D6FB4", "#27406B", "#C0392B", "#222222"

ORDER = [("base_no_rlhf", "Qwen3.5-9B-Base\n(no FT)"),
         ("instruct_base", "Qwen3.5-9B\n(no adapter)"),
         ("sft_9b", "+ SFT v4"),
         ("dpo_9b", "+ DPO v5")]


def load():
    out = []
    for key, lab in ORDER:
        d = json.loads(Path(f"outputs/eval_9b_{key}_n30.json").read_text())
        n = d["total"]
        out.append({"key": key, "label": lab, "n": n,
                    "schema": d["schema_pass"] / n * 100,
                    "content": d["content_pass"] / n * 100, "raw": d})
    return out


def chart(rows):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(rows)); w = 0.36
    sch = [r["schema"] for r in rows]; con = [r["content"] for r in rows]
    ax.bar(x - w/2, sch, w, label="스키마 준수 통과 (포맷 요구)", color=GRAY, zorder=3)
    ax.bar(x + w/2, con, w, label="내용만 통과 (포맷 무시, 추론만)", color=BLUE, zorder=3)
    for xi, v in zip(x - w/2, sch):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=11.5, color=TEXT)
    for xi, v in zip(x + w/2, con):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=11.5, fontweight="bold", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in rows], fontsize=11)
    ax.set_ylim(0, 110); ax.set_ylabel("통과율 (%)", fontsize=12)
    ax.set_title("Qwen3.5-9B 포맷(스키마) vs 추론(내용) 통과율 (n=30)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#CCC"); ax.spines["bottom"].set_color("#CCC")
    ax.tick_params(length=0); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EAEAEA")
    ax.annotate("0% → 47%\n추론은 했지만\n포맷만 틀림", xy=(1 + w/2, 46.7), xytext=(1.55, 72),
                ha="center", fontsize=10.5, fontweight="bold", color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#EAF1F9", ec=BLUE, alpha=0.9))
    plt.tight_layout()
    out = ASSETS / "chart_9b_schema_vs_content.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    rows = load()
    print(f"{'모델':22}{'schema':>9}{'content':>9}")
    for r in rows:
        print(f"{r['label'].replace(chr(10),' '):22}{r['schema']:>8.1f}%{r['content']:>8.1f}%")
    chart(rows)
