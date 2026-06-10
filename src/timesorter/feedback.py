"""사용자 피드백 → DPO 학습 데이터 변환.

발표 자료 STEP 5-6의 미구현 구간: 사용자가 모델의 우선순위를 수정하면
(모델 출력 = rejected, 수정 반영 출력 = chosen) DPO 쌍을 생성해 누적한다.

저장 형식은 data/dpo_pairs_v3.parquet과 동일 컬럼(prompt/chosen/rejected/persona/today/category)
이므로 누적 피드백을 기존 쌍과 병합해 바로 재학습에 쓸 수 있다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data.schema import parse_lenient


@dataclass
class FeedbackRecord:
    prompt: str                 # 사용자 입력 (할 일 목록 원문)
    model_output: str           # 모델이 출력한 v2/v3 JSON 원문
    corrected_order: list[int]  # 사용자가 수정한 task_id 순서
    persona: str = "직장인"
    today: str = ""
    note: str = ""              # 선택: 수정 이유 메모
    extra: dict = field(default_factory=dict)


def to_dpo_pair(record: FeedbackRecord) -> dict | None:
    """피드백 1건 → DPO 쌍. 순서 변경이 없거나 파싱 불가면 None."""
    resp = parse_lenient(record.model_output)
    if resp is None or not resp.tasks:
        return None

    task_ids = {t.id for t in resp.tasks}
    if set(record.corrected_order) != task_ids:
        raise ValueError(
            f"corrected_order={record.corrected_order}가 태스크 id 집합 {sorted(task_ids)}과 불일치"
        )
    if record.corrected_order == resp.priority_order:
        return None  # 수정 없음 — 학습 신호 아님

    chosen = resp.model_copy(update={"priority_order": record.corrected_order})
    return {
        "prompt": record.prompt,
        "chosen": chosen.model_dump_json(),
        "rejected": resp.model_dump_json(),
        "persona": record.persona,
        "today": record.today,
        "category": "user_feedback",
    }


def append_feedback(record: FeedbackRecord, path: str | Path = "data/feedback.jsonl") -> None:
    """피드백 레코드를 jsonl에 누적 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def export_dpo_pairs(
    feedback_path: str | Path = "data/feedback.jsonl",
    out_path: str | Path = "data/dpo_pairs_feedback.parquet",
) -> int:
    """누적 피드백 jsonl → DPO 쌍 parquet. 생성된 쌍 수 반환."""
    import pandas as pd

    fp = Path(feedback_path)
    if not fp.exists():
        return 0

    pairs: list[dict] = []
    with fp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = FeedbackRecord(**json.loads(line))
            pair = to_dpo_pair(rec)
            if pair:
                pairs.append(pair)

    if pairs:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pairs).to_parquet(str(out), index=False)
    return len(pairs)
