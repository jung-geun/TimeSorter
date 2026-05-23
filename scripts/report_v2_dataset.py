#!/usr/bin/env python
"""Phase G: v2 데이터셋 머지 + 통계 보고서.

SFT 소스:
  data/scheduler_v2_regen.parquet        (Phase D)
  data/scheduler_v2_nemotron_extra.parquet (Phase C)
  data/orca_ko_filtered_dpo.parquet      (Phase E)
  data/xlam_ko_scheduled.parquet         (Phase E)
  data/refusals_sft_v2.parquet           (Phase B)

DPO 소스:
  data/dpo_pairs_v2.parquet              (Phase F, Phase B 포함)

산출물:
  data/scheduler_v2_combined.parquet     (SFT 학습 입력)
  docs/dataset_analysis_v2.md            (보고서)

사용:
  uv run python scripts/report_v2_dataset.py
  uv run python scripts/report_v2_dataset.py --no-merge   # 통계만
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.data.schema import parse_lenient

_SFT_SOURCES = [
    ("data/scheduler_v2_regen.parquet",         "regen"),
    ("data/scheduler_v2_nemotron_extra.parquet", "nemotron_extra"),
    ("data/orca_ko_filtered_dpo.parquet",        "orca_ko"),
    ("data/xlam_ko_scheduled.parquet",           "xlam_ko"),
    ("data/refusals_sft_v2.parquet",             "refusal"),
]

_DPO_SOURCE = "data/dpo_pairs_v2.parquet"

_SFT_OUT   = "data/scheduler_v2_combined.parquet"
_REPORT_OUT = "docs/dataset_analysis_v2.md"


def _parse_rate(df: pd.DataFrame, col: str = "chosen") -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    passed = df[col].apply(lambda x: parse_lenient(str(x)) is not None).sum()
    return passed / len(df)


def _score_stats(df: pd.DataFrame) -> dict:
    """chosen 컬럼에서 scores를 파싱해 4축 평균/표준편차 계산."""
    axes = ["urgency", "importance", "dependency", "time_constraint"]
    records: list[dict] = []
    for val in df.get("chosen", []):
        parsed = parse_lenient(str(val))
        if parsed is None:
            continue
        for s in parsed.scores:
            records.append({
                "urgency": s.urgency, "importance": s.importance,
                "dependency": s.dependency, "time_constraint": s.time_constraint,
            })
    if not records:
        return {}
    stats_df = pd.DataFrame(records)
    result = {}
    for ax in axes:
        result[ax] = {
            "mean": round(stats_df[ax].mean(), 2),
            "std": round(stats_df[ax].std(), 2),
            "dist": stats_df[ax].value_counts().sort_index().to_dict(),
        }
    return result


def merge_sft(no_merge: bool) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for path, source_tag in _SFT_SOURCES:
        p = Path(path)
        if not p.exists():
            print(f"[스킵] {path} — 없음")
            continue
        df = pd.read_parquet(path)
        if "source" not in df.columns:
            df["source"] = source_tag
        frames.append(df)
        print(f"[SFT] {path}: {len(df)}행  parse_rate={_parse_rate(df):.1%}")

    if not frames:
        print("[경고] SFT 소스 없음")
        return None

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["prompt"])
    print(f"[SFT] 중복 제거: {before}→{len(combined)}행")

    if not no_merge:
        Path(_SFT_OUT).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(_SFT_OUT, index=False)
        print(f"[SFT] 저장 → {_SFT_OUT}  ({len(combined)}행)")

    return combined


def load_dpo() -> pd.DataFrame | None:
    p = Path(_DPO_SOURCE)
    if not p.exists():
        print(f"[스킵] {_DPO_SOURCE} — 없음")
        return None
    df = pd.read_parquet(str(p))
    print(f"[DPO] {_DPO_SOURCE}: {len(df)}행  chosen_parse={_parse_rate(df):.1%}")
    return df


def write_report(sft_df: pd.DataFrame | None, dpo_df: pd.DataFrame | None) -> None:
    lines: list[str] = ["# v2 데이터셋 분석 보고서\n"]

    # SFT 통계
    if sft_df is not None:
        lines.append("## SFT 데이터\n")
        lines.append(f"- 총 행 수: **{len(sft_df)}**\n")
        lines.append(f"- chosen parse_lenient 통과율: **{_parse_rate(sft_df):.1%}**\n")

        if "source" in sft_df.columns:
            lines.append("\n### 소스별 분포\n")
            lines.append("| 소스 | 행 수 | 비율 |\n|------|-------|------|\n")
            for src, cnt in sft_df["source"].value_counts().items():
                lines.append(f"| {src} | {cnt} | {cnt/len(sft_df):.1%} |\n")

        if "persona" in sft_df.columns:
            lines.append("\n### 페르소나 분포 (상위 15)\n")
            lines.append("| 페르소나 | 행 수 |\n|---------|------|\n")
            for persona, cnt in sft_df["persona"].value_counts().head(15).items():
                lines.append(f"| {persona} | {cnt} |\n")

        stats = _score_stats(sft_df)
        if stats:
            lines.append("\n### 4축 점수 통계\n")
            lines.append("| 축 | 평균 | 표준편차 | 경고 |\n|----|------|----------|------|\n")
            for ax, s in stats.items():
                warn = ""
                if s["std"] < 1.0:
                    warn = "⚠️ 편향 (std<1)"
                lines.append(f"| {ax} | {s['mean']} | {s['std']} | {warn} |\n")

    # DPO 통계
    if dpo_df is not None:
        lines.append("\n## DPO 데이터\n")
        lines.append(f"- 총 쌍 수: **{len(dpo_df)}**\n")
        lines.append(f"- chosen parse_lenient 통과율: **{_parse_rate(dpo_df):.1%}**\n")

        if "category" in dpo_df.columns:
            lines.append("\n### rejected 카테고리 분포\n")
            lines.append("| 카테고리 | 쌍 수 | 비율 |\n|---------|-------|------|\n")
            for cat, cnt in dpo_df["category"].value_counts().items():
                lines.append(f"| {cat} | {cnt} | {cnt/len(dpo_df):.1%} |\n")

        if "persona" in dpo_df.columns:
            lines.append("\n### 페르소나 분포 (상위 10)\n")
            lines.append("| 페르소나 | 쌍 수 |\n|---------|------|\n")
            for persona, cnt in dpo_df["persona"].value_counts().head(10).items():
                lines.append(f"| {persona} | {cnt} |\n")

    Path(_REPORT_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(_REPORT_OUT).write_text("".join(lines), encoding="utf-8")
    print(f"\n[보고서] {_REPORT_OUT}")
    # 콘솔에도 핵심만 출력
    print("".join(lines[:30]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-merge", action="store_true", help="parquet 저장 없이 통계만")
    args = parser.parse_args()

    sft_df = merge_sft(args.no_merge)
    dpo_df = load_dpo()
    write_report(sft_df, dpo_df)


if __name__ == "__main__":
    main()
