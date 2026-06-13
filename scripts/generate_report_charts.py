#!/usr/bin/env python3
"""TimeSorter 종합 벤치마크 리포트 — 차트 + 테이블 + 응답 예시.

생성 파일:
  outputs/report/chart_overall.png       전체 통과율 진행 막대
  outputs/report/chart_scenario.png      시나리오별 히트맵
  outputs/report/chart_violation.png     위반 유형 스택 막대
  outputs/report/chart_qwen35.png        Qwen3.5 전용 비교
  outputs/report/table_results.csv       전체 수치 테이블
  outputs/report/table_results.md        Markdown 테이블

사용:
  uv run python scripts/generate_report_charts.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("outputs/report")
OUT.mkdir(parents=True, exist_ok=True)

# ── 데이터 정의 ──────────────────────────────────────────────────────────────
CHECKPOINTS = [
    # (key, label, base_model, stage, eval_file, manual)
    ("dpo_v3",    "DPO v3\nQwen3-4B",        "Qwen3-4B",       "DPO",   "outputs/eval_dpo_v3.json",  None),
    ("grpo_v4",   "GRPO v4\nQwen3-4B",       "Qwen3-4B",       "GRPO",  "outputs/eval_grpo_v4.json", None),
    ("sft_v4",    "SFT v4\nQwen3-4B",        "Qwen3-4B",       "SFT",   "outputs/eval_sft_v4.json",  None),
    ("dpo_v5",    "DPO v5\nQwen3-4B",        "Qwen3-4B",       "DPO",   "outputs/eval_dpo_v5.json",  None),
    ("q35_none",  "No adapter\nQwen3.5-4B",  "Qwen3.5-4B",     "BASE",  "outputs/eval_qwen35_instruct_base_n30.json", None),
    ("q35_base",  "No adapter\nQwen3.5-Base","Qwen3.5-4B-Base","BASE",  "outputs/eval_qwen35_base_no_rlhf_n30.json",  None),
    ("q35_sft",   "SFT v4\nQwen3.5-4B",     "Qwen3.5-4B",     "SFT",   "outputs/eval_qwen35_sft_q35_n30.json", None),
    ("q35_dpo",   "DPO v5\nQwen3.5-4B",     "Qwen3.5-4B",     "DPO",   None, {
        "clean": 27, "total": 30,
        "per_scenario": {
            "dated_mixed":      {"n": 9,  "violated": 0},
            "intraday":         {"n": 6,  "violated": 0},
            "risk":             {"n": 5,  "violated": 1},
            "relative":         {"n": 4,  "violated": 0},
            "dependency_chain": {"n": 6,  "violated": 2},
        },
        "per_kind": {"chain": 4, "past_rank": 1},
    }),
]

SCENARIOS = ["dated_mixed", "intraday", "risk", "relative", "dependency_chain"]
SC_SHORT   = ["Dated\nMixed", "Intraday", "Risk", "Relative", "Dep.\nChain"]

MODEL_COLORS = {
    "Qwen3-4B":       "#4C72B0",
    "Qwen3.5-4B":     "#DD8452",
    "Qwen3.5-4B-Base":"#55A868",
}
STAGE_HATCHES = {"SFT": "///", "DPO": "xxx", "GRPO": "...", "BASE": ""}


def load_all() -> list[dict]:
    rows = []
    for key, label, base, stage, fpath, manual in CHECKPOINTS:
        if fpath and Path(fpath).exists():
            d = json.loads(Path(fpath).read_text())
        elif manual:
            d = manual
        else:
            continue
        overall = d["clean"] / d["total"] * 100
        per_sc = {}
        for sc, v in d["per_scenario"].items():
            n = v["n"]; viol = v.get("violated", 0)
            per_sc[sc] = (n - viol) / n * 100 if n else 0.0
        rows.append({
            "key": key, "label": label, "base": base, "stage": stage,
            "overall": overall, "n": d["total"],
            "per_sc": per_sc,
            "per_kind": d.get("per_kind", {}),
        })
    return rows


# ── Chart 1: 전체 통과율 진행 ────────────────────────────────────────────────
def chart_overall(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(rows))

    for i, r in enumerate(rows):
        color  = MODEL_COLORS.get(r["base"], "#999")
        hatch  = STAGE_HATCHES.get(r["stage"], "")
        bar = ax.bar(i, r["overall"], color=color, hatch=hatch, alpha=0.88,
                     edgecolor="black", linewidth=0.8, width=0.65)
        ax.text(i, r["overall"] + 0.8, f"{r['overall']:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i, -5, f"n={r['n']}", ha="center", va="top",
                fontsize=7, color="#555")

    # 주요 마일스톤 주석
    milestones = {
        "sft_v4":  ("SFT 데이터\n품질 개선\n+loss masking", 77.3),
        "q35_sft": ("Qwen3.5\n모델 업그레이드", 90.0),
    }
    for key, (note, y) in milestones.items():
        idx = next((i for i, r in enumerate(rows) if r["key"] == key), None)
        if idx is not None:
            ax.annotate(note, xy=(idx, y), xytext=(idx + 0.5, y + 8),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1),
                        fontsize=8, color="#333",
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="#ccc"))

    ax.axhline(90, color="red",    ls="--", lw=0.8, alpha=0.5, label="90% target")
    ax.axhline(77.3, color="orange", ls="--", lw=0.8, alpha=0.5, label="SFT v4 baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_title("TimeSorter — Overall Pass Rate by Checkpoint", fontsize=13, fontweight="bold")
    ax.set_ylim(-10, 115)
    ax.grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(facecolor=MODEL_COLORS["Qwen3-4B"],       label="Qwen3-4B-Instruct"),
        mpatches.Patch(facecolor=MODEL_COLORS["Qwen3.5-4B"],     label="Qwen3.5-4B"),
        mpatches.Patch(facecolor=MODEL_COLORS["Qwen3.5-4B-Base"],label="Qwen3.5-4B-Base"),
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="black", label="SFT"),
        mpatches.Patch(facecolor="white", hatch="xxx", edgecolor="black", label="DPO"),
        mpatches.Patch(facecolor="white", hatch="...", edgecolor="black", label="GRPO"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left", ncol=2)

    plt.tight_layout()
    path = OUT / "chart_overall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── Chart 2: 시나리오별 히트맵 ──────────────────────────────────────────────
def chart_scenario_heatmap(rows: list[dict]) -> None:
    labels = [r["label"].replace("\n", " ") for r in rows]
    matrix = np.array([[r["per_sc"].get(sc, 0.0) for sc in SCENARIOS] for r in rows])

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.55 + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(SC_SHORT, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Per-Scenario Pass Rate (%)", fontsize=12, fontweight="bold")

    for i in range(len(rows)):
        for j in range(len(SCENARIOS)):
            val = matrix[i, j]
            color = "white" if val < 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Pass Rate (%)", shrink=0.8)
    plt.tight_layout()
    path = OUT / "chart_scenario.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── Chart 3: 위반 유형 스택 막대 ────────────────────────────────────────────
def chart_violations(rows: list[dict]) -> None:
    viol_types = ["past_rank", "intraday_order", "chain", "risk_importance", "none_first", "parse_or_count"]
    viol_colors = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#a9a9a9"]
    viol_labels = ["Past Rank", "Intraday Order", "Chain", "Risk", "None First", "Parse Error"]

    labels = [r["label"].replace("\n", " ") for r in rows]
    data = {v: [r["per_kind"].get(v, 0) for r in rows] for v in viol_types}

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for vtype, color, vlabel in zip(viol_types, viol_colors, viol_labels):
        vals = np.array(data[vtype], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=vlabel,
               alpha=0.85, edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(i, b + v / 2, str(int(v)), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Violation Count", fontsize=10)
    ax.set_title("Violation Types by Checkpoint", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUT / "chart_violation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── Chart 4: Qwen3.5 전용 비교 (2-panel) ────────────────────────────────────
def chart_qwen35(rows: list[dict]) -> None:
    q35_rows = [r for r in rows if "Qwen3.5" in r["base"]]
    if not q35_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Qwen3.5-4B: Base Model vs Fine-tuned", fontsize=13, fontweight="bold")

    # Left: overall
    ax1 = axes[0]
    x = np.arange(len(q35_rows))
    for i, r in enumerate(q35_rows):
        color = MODEL_COLORS.get(r["base"], "#999")
        hatch = STAGE_HATCHES.get(r["stage"], "")
        ax1.bar(i, r["overall"], color=color, hatch=hatch, alpha=0.88,
                edgecolor="black", linewidth=0.8, width=0.6)
        ax1.text(i, r["overall"] + 1, f"{r['overall']:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(i, -5, f"n={r['n']}", ha="center", fontsize=7, color="#555")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["label"] for r in q35_rows], fontsize=8)
    ax1.set_ylim(-10, 110)
    ax1.set_ylabel("Pass Rate (%)")
    ax1.set_title("Overall Pass Rate")
    ax1.axhline(90, color="red", ls="--", lw=0.8, alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # Right: per-scenario grouped
    ax2 = axes[1]
    n_m = len(q35_rows); n_sc = len(SCENARIOS)
    w = 0.8 / n_m
    for j, r in enumerate(q35_rows):
        offsets = np.arange(n_sc) + (j - n_m/2 + 0.5) * w
        vals = [r["per_sc"].get(sc, 0) for sc in SCENARIOS]
        color = MODEL_COLORS.get(r["base"], "#999")
        hatch = STAGE_HATCHES.get(r["stage"], "")
        ax2.bar(offsets, vals, width=w*0.9, color=color, hatch=hatch,
                alpha=0.88, edgecolor="black", linewidth=0.5,
                label=r["label"].replace("\n", " "))
    ax2.set_xticks(np.arange(n_sc))
    ax2.set_xticklabels(SC_SHORT, fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Pass Rate (%)")
    ax2.set_title("Per-Scenario Breakdown")
    ax2.legend(fontsize=7, loc="upper left", ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(facecolor=MODEL_COLORS["Qwen3.5-4B"],      label="Qwen3.5-4B"),
        mpatches.Patch(facecolor=MODEL_COLORS["Qwen3.5-4B-Base"], label="Qwen3.5-4B-Base"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    path = OUT / "chart_qwen35.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── Table: CSV + Markdown ────────────────────────────────────────────────────
def save_tables(rows: list[dict]) -> None:
    headers = ["model", "base_model", "stage", "n_samples", "overall_pct",
               "dated_mixed", "intraday", "risk", "relative", "dependency_chain",
               "viol_past_rank", "viol_intraday_order", "viol_chain",
               "viol_risk_importance", "viol_none_first", "viol_parse"]

    csv_path = OUT / "table_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            vk = r["per_kind"]
            w.writerow([
                r["label"].replace("\n", " "), r["base"], r["stage"], r["n"],
                f"{r['overall']:.1f}",
                *[f"{r['per_sc'].get(sc, 0.0):.1f}" for sc in SCENARIOS],
                vk.get("past_rank", 0), vk.get("intraday_order", 0),
                vk.get("chain", 0), vk.get("risk_importance", 0),
                vk.get("none_first", 0), vk.get("parse_or_count", 0),
            ])
    print(f"[saved] {csv_path}")

    # Markdown
    md_path = OUT / "table_results.md"
    lines = [
        "# TimeSorter 벤치마크 결과 테이블\n",
        "## 전체 통과율 및 시나리오별 통과율\n",
        "| 모델 | Base | Stage | N | 전체 | Dated | Intraday | Risk | Relative | Chain |",
        "|------|------|-------|---|------|-------|----------|------|----------|-------|",
    ]
    for r in rows:
        sc = r["per_sc"]
        lines.append(
            f"| {r['label'].replace(chr(10), ' ')} "
            f"| {r['base']} | {r['stage']} | {r['n']} "
            f"| **{r['overall']:.1f}%** "
            f"| {sc.get('dated_mixed',0):.0f}% "
            f"| {sc.get('intraday',0):.0f}% "
            f"| {sc.get('risk',0):.0f}% "
            f"| {sc.get('relative',0):.0f}% "
            f"| {sc.get('dependency_chain',0):.0f}% |"
        )

    lines += [
        "\n## 위반 유형별 건수\n",
        "| 모델 | Stage | past_rank | intraday_order | chain | risk | none_first | parse |",
        "|------|-------|-----------|----------------|-------|------|------------|-------|",
    ]
    for r in rows:
        vk = r["per_kind"]
        lines.append(
            f"| {r['label'].replace(chr(10), ' ')} | {r['stage']} "
            f"| {vk.get('past_rank',0)} | {vk.get('intraday_order',0)} "
            f"| {vk.get('chain',0)} | {vk.get('risk_importance',0)} "
            f"| {vk.get('none_first',0)} | {vk.get('parse_or_count',0)} |"
        )

    lines += [
        "\n## 메모\n",
        "- n=150: held-out 150샘플 전체 평가 (Qwen3-4B 체크포인트)",
        "- n=30: 30샘플 빠른 평가 (Qwen3.5-4B 체크포인트)",
        "- DPO v5 (Qwen3.5-4B): 이전 세션 local eval 결과 (27/30=90.0%)",
        "- Qwen3.5-4B no adapter: thinking mode 출력 → parse 전부 실패 (0.0%)",
        "- Qwen3.5-4B-Base no adapter: RLHF 없어 형식 불완전하나 일부 통과 (16.7%)",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {md_path}")


# ── 응답 예시 수집 스크립트 안내 ─────────────────────────────────────────────
def save_example_runner_note() -> None:
    note = OUT / "EXAMPLES_TODO.md"
    note.write_text(
        "# 응답 예시 수집\n\n"
        "GPU 학습 완료 후 아래 명령으로 실행 (raw 출력 저장 포함):\n\n"
        "```bash\n"
        "uv run python scripts/benchmark_qwen35.py \\\n"
        "  --target sft_q35,dpo_q35 --limit 10 --force\n"
        "```\n\n"
        "benchmark_qwen35.py에 `save_raw=True` 옵션 추가 후 실행하면\n"
        "`outputs/eval_qwen35_*_n10.json`의 details에 raw 출력이 포함됩니다.\n\n"
        "## 현재 알려진 위반 유형별 예시 패턴\n\n"
        "### chain 위반 (dependency_chain 시나리오)\n"
        "- 원인: A→B→C 의존성에서 B가 A보다 상위에 배치\n"
        "- 모델 출력 예: `[C, B, A, ...]` 또는 `[B, C, A, ...]` 형태로 잘못된 순서\n\n"
        "### past_rank 위반 (dated_mixed 시나리오)\n"
        "- 원인: 지난 날짜의 태스크를 미래 태스크보다 상위에 배치\n"
        "- 모델 출력 예: urgency/time_constraint 점수를 날짜 무관하게 높게 부여\n",
        encoding="utf-8",
    )
    print(f"[saved] {note}")


if __name__ == "__main__":
    rows = load_all()
    print(f"Loaded {len(rows)} checkpoints")
    for r in rows:
        print(f"  {r['label'].replace(chr(10),' '):<35} {r['overall']:.1f}%  (n={r['n']})")

    chart_overall(rows)
    chart_scenario_heatmap(rows)
    chart_violations(rows)
    chart_qwen35(rows)
    save_tables(rows)
    save_example_runner_note()
    print(f"\nAll outputs → {OUT}/")
