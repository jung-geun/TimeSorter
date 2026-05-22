#!/usr/bin/env python
"""한국어 데이터셋 분석 리포트를 docs/dataset_analysis.md 로 생성합니다."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 마크다운 표 헬퍼
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    """헤더 + 행 리스트를 마크다운 표 문자열로 변환합니다."""
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 섹션 생성 함수
# ---------------------------------------------------------------------------

def section_overview(sched_df: pd.DataFrame, dpo_df: pd.DataFrame) -> str:
    def missing(df: pd.DataFrame) -> int:
        return int(df.isnull().sum().sum())

    rows = [
        [
            "`data/scheduler_ko.parquet`",
            len(sched_df),
            ", ".join(sched_df.columns),
            missing(sched_df),
            "SFT Chosen 후보",
        ],
        [
            "`data/dpo_pairs.parquet`",
            len(dpo_df),
            ", ".join(dpo_df.columns),
            missing(dpo_df),
            "DPO 학습 페어",
        ],
    ]
    table = _md_table(["데이터셋", "행 수", "컬럼", "결측치", "용도"], rows)
    return "## 1. 개요\n\n" + table


def section_persona(sched_df: pd.DataFrame) -> str:
    persona_pattern = re.compile(r"(.+)\s*\((.+),\s*(\d+)세\)")
    ages, occupations = [], []
    for p_str in sched_df["persona"]:
        m = persona_pattern.search(str(p_str))
        if m:
            _, occ, age = m.groups()
            ages.append(int(age))
            occupations.append(occ.strip())

    ages_arr = np.array(ages)
    summary = (
        f"총 {len(ages_arr)}명 파싱 완료 | "
        f"평균 {ages_arr.mean():.1f}세 | "
        f"최소 {ages_arr.min()}세 | "
        f"최대 {ages_arr.max()}세"
    )

    # 연령대 분포
    age_groups = Counter((ages_arr // 10) * 10)
    age_rows = []
    for decade in sorted(age_groups):
        cnt = age_groups[decade]
        age_rows.append([f"{decade}대", cnt, f"{cnt / len(ages_arr) * 100:.1f}%"])
    age_table = _md_table(["연령대", "인원 (명)", "비율"], age_rows)

    # 직업 Top 10
    occ_counts = Counter(occupations)
    occ_rows = []
    for occ, cnt in occ_counts.most_common(10):
        occ_rows.append([occ, cnt, f"{cnt / len(occupations) * 100:.1f}%"])
    occ_table = _md_table(["직업", "인원 (명)", "비율"], occ_rows)

    return (
        "## 2. 페르소나 다양성 (scheduler_ko)\n\n"
        + summary + "\n\n"
        + "### 연령대 분포\n\n" + age_table + "\n\n"
        + "### 직업 Top 10\n\n" + occ_table
    )


def section_length(sched_df: pd.DataFrame) -> str:
    stats = {
        "prompt 문자 수": sched_df["prompt"].str.len(),
        "chosen 문자 수": sched_df["chosen"].str.len(),
        "prompt 줄 수": sched_df["prompt"].str.count("\n") + 1,
        "chosen 줄 수": sched_df["chosen"].str.count("\n") + 1,
    }

    stat_keys = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    header = ["항목"] + stat_keys
    rows = []
    for name, series in stats.items():
        desc = series.describe()
        rows.append([name] + [f"{desc[k]:.1f}" for k in stat_keys])

    return "## 3. 길이 분포 (scheduler_ko)\n\n" + _md_table(header, rows)


def section_format(sched_df: pd.DataFrame) -> str:
    pattern = re.compile(r"^\d+\)\s+.+?\s+-\s+.+$")
    fail_indices = []

    for idx, row in sched_df.iterrows():
        lines = [l.strip() for l in row["chosen"].split("\n") if l.strip()]
        valid = sum(1 for l in lines if pattern.match(l))
        if len(lines) == 0 or valid < len(lines):
            fail_indices.append(idx)

    total = len(sched_df)
    passed = total - len(fail_indices)
    rate = passed / total * 100

    table = _md_table(
        ["전체", "통과", "실패", "통과율"],
        [[total, passed, len(fail_indices), f"{rate:.1f}%"]],
    )

    body = "## 4. 포맷 준수율 (scheduler_ko)\n\n" + table
    if fail_indices:
        body += "\n\n### 실패 샘플 인덱스\n\n" + ", ".join(str(i) for i in fail_indices)
    return body


def section_dpo(dpo_df: pd.DataFrame) -> str:
    # 페어 타입 분포
    pair_rows = []
    for pair, cnt in dpo_df["pair"].value_counts().items():
        pair_rows.append([pair, cnt, f"{cnt / len(dpo_df) * 100:.1f}%"])
    pair_table = _md_table(["페어 타입", "개수", "비율"], pair_rows)

    # chosen vs rejected 길이 비교
    dpo_df = dpo_df.copy()
    dpo_df["chosen_len"] = dpo_df["chosen"].str.len()
    dpo_df["rejected_len"] = dpo_df["rejected"].str.len()

    len_rows = [
        ["chosen", int(dpo_df["chosen_len"].mean()), int(dpo_df["chosen_len"].median()), int(dpo_df["chosen_len"].max())],
        ["rejected", int(dpo_df["rejected_len"].mean()), int(dpo_df["rejected_len"].median()), int(dpo_df["rejected_len"].max())],
    ]
    len_table = _md_table(["응답", "평균 문자 수", "중앙값", "최대"], len_rows)

    unique_personas = dpo_df["persona"].nunique()

    return (
        "## 5. DPO 페어 통계 (dpo_pairs)\n\n"
        + f"고유 페르소나 수: **{unique_personas}명**\n\n"
        + "### 페어 타입 분포\n\n" + pair_table + "\n\n"
        + "### Chosen vs Rejected 길이 비교\n\n" + len_table
    )


def section_samples(sched_df: pd.DataFrame, dpo_df: pd.DataFrame) -> str:
    rng = np.random.default_rng(42)

    # scheduler_ko 샘플 2개
    sched_idx = rng.choice(len(sched_df), size=2, replace=False)
    sched_meta_rows = []
    sched_details = []
    for i, idx in enumerate(sched_idx, 1):
        row = sched_df.iloc[idx]
        sched_meta_rows.append([i, row.get("persona", ""), row.get("source", ""), int(row.get("original_idx", idx))])
        block = (
            f"<details>\n<summary>샘플 {i} — {row.get('persona', '')}</summary>\n\n"
            f"**Prompt**\n\n```\n{row['prompt']}\n```\n\n"
            f"**Chosen**\n\n```\n{row['chosen']}\n```\n\n"
            "</details>"
        )
        sched_details.append(block)

    sched_meta_table = _md_table(["#", "페르소나", "소스", "original_idx"], sched_meta_rows)

    # dpo_pairs 샘플 2개
    dpo_idx = rng.choice(len(dpo_df), size=min(2, len(dpo_df)), replace=False)
    dpo_meta_rows = []
    dpo_details = []
    for i, idx in enumerate(dpo_idx, 1):
        row = dpo_df.iloc[idx]
        dpo_meta_rows.append([i, row.get("persona", ""), row.get("pair", "")])
        block = (
            f"<details>\n<summary>DPO 샘플 {i} — {row.get('persona', '')}</summary>\n\n"
            f"**Prompt**\n\n```\n{row['prompt']}\n```\n\n"
            f"**Chosen**\n\n```\n{row['chosen']}\n```\n\n"
            f"**Rejected**\n\n```\n{row['rejected']}\n```\n\n"
            "</details>"
        )
        dpo_details.append(block)

    dpo_meta_table = _md_table(["#", "페르소나", "페어 타입"], dpo_meta_rows)

    return (
        "## 6. 샘플 미리보기\n\n"
        + "### scheduler_ko 샘플 (2개)\n\n"
        + sched_meta_table + "\n\n"
        + "\n\n".join(sched_details) + "\n\n"
        + "### dpo_pairs 샘플 (2개)\n\n"
        + dpo_meta_table + "\n\n"
        + "\n\n".join(dpo_details)
    )


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    sched_df = pd.read_parquet("data/scheduler_ko.parquet")
    dpo_df = pd.read_parquet("data/dpo_pairs.parquet")

    print(f"scheduler_ko: {len(sched_df)}행  |  dpo_pairs: {len(dpo_df)}행")

    header = (
        "# 한국어 데이터셋 분석 리포트\n\n"
        "> 생성 스크립트: `scripts/gen_dataset_report.py`  \n"
        f"> 대상: `data/scheduler_ko.parquet` (SFT) + `data/dpo_pairs.parquet` (DPO)"
    )

    sections = [
        section_overview(sched_df, dpo_df),
        section_persona(sched_df),
        section_length(sched_df),
        section_format(sched_df),
        section_dpo(dpo_df),
        section_samples(sched_df, dpo_df),
    ]

    md = header + "\n\n" + "\n\n".join(sections) + "\n"
    out_path = Path("docs/dataset_analysis.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"리포트 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
