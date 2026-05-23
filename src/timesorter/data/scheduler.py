from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset

from .schema import SCHEDULER_SYSTEM_PROMPT_V1, SCHEDULER_SYSTEM_PROMPT_V2, render_system_prompt

if TYPE_CHECKING:
    pass

_PERSONAS = ["직장인", "학생", "프리랜서", "부모"]

# 하위 호환 — v1 어댑터 학습 코드가 직접 참조하는 경우 대비
_SYSTEM_TMPL = SCHEDULER_SYSTEM_PROMPT_V1

_KO_SCHEDULING_KEYWORDS = [
    "일정", "우선순위", "할 일", "할일", "계획", "스케줄", "업무",
    "마감", "기한", "태스크", "task",
]


def _to_chatml(
    prompt: str,
    response: str,
    persona: str = "직장인",
    schema_version: str = "v1",
) -> dict:
    system_tmpl = (
        SCHEDULER_SYSTEM_PROMPT_V2 if schema_version == "v2" else SCHEDULER_SYSTEM_PROMPT_V1
    )
    return {
        "messages": [
            {"role": "system", "content": render_system_prompt(system_tmpl, persona)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def _load_ko_ultrafeedback_scheduling(n: int) -> list[dict]:
    """ko_Ultrafeedback에서 스케줄링 키워드 샘플 n개를 ChatML 형식으로 반환."""
    from datasets import load_dataset as _load
    local_path = Path("data/ko_Ultrafeedback_binarized.parquet")
    if local_path.exists():
        uf = Dataset.from_parquet(str(local_path))
    else:
        uf = _load("maywell/ko_Ultrafeedback_binarized", split="train")

    kw = _KO_SCHEDULING_KEYWORDS

    def _is_scheduling(row: dict) -> bool:
        return any(k in row["prompt"].lower() for k in kw)

    filtered = uf.filter(_is_scheduling, desc="scheduling 키워드 필터")
    print(f"[scheduler] ko_Ultrafeedback 스케줄링 필터: {len(filtered)}개 → {n}개 사용")
    rows = []
    for row in filtered.select(range(min(n, len(filtered)))):
        rows.append(_to_chatml(row["prompt"], row["chosen"]))
    return rows


def load_scheduler_dataset(
    parquet_path: str | None = None,
    ko_ultrafeedback_n: int = 0,
    max_samples: int | None = None,
    schema_version: str = "v1",
) -> Dataset:
    """SFT용 스케줄러 데이터셋 로드.

    Args:
        parquet_path: scheduler_ko.parquet 경로. 존재하면 우선 로드.
        ko_ultrafeedback_n: ko_Ultrafeedback 스케줄링 샘플 혼합 수.
                            0이면 미사용. parquet가 없을 때 자동으로 500으로 설정됨.
        max_samples: 최종 반환 샘플 수 상한.
    """
    rows: list[dict] = []

    if parquet_path and Path(parquet_path).exists():
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        for _, r in df.iterrows():
            persona = r.get("persona", "직장인")
            rows.append(_to_chatml(str(r["prompt"]), str(r["chosen"]), persona, schema_version))
        print(f"[scheduler] parquet 로드: {len(rows)}개 ({parquet_path})")

    # parquet가 없거나 비어있으면 ko_Ultrafeedback으로 자동 fallback
    if not rows and ko_ultrafeedback_n == 0:
        ko_ultrafeedback_n = 500

    if ko_ultrafeedback_n > 0:
        rows.extend(_load_ko_ultrafeedback_scheduling(ko_ultrafeedback_n))

    if max_samples is not None:
        rows = rows[:max_samples]

    ds = Dataset.from_list(rows)
    print(f"[scheduler] 최종 SFT 데이터: {len(ds)}개")
    return ds
