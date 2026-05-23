from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset

from .schema import SCHEDULER_SYSTEM_PROMPT_V1, SCHEDULER_SYSTEM_PROMPT_V2, render_system_prompt

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

_REQUIRED_COLS = {"prompt", "chosen", "rejected"}


def _apply_system_to_dpo(
    ds: Dataset,
    tokenizer,
    schema_version: str = "v1",
    persona_col: str = "persona",
) -> Dataset:
    """DPO 데이터셋의 prompt를 system+user chat template 형식으로 변환.

    TRL DPOTrainer는 prompt/chosen/rejected가 이미 chat template이 적용된 문자열이거나
    messages list여야 한다. 이 함수는 raw text prompt에 system prompt를 씌운
    messages list 형식으로 변환한다.
    """
    system_tmpl = (
        SCHEDULER_SYSTEM_PROMPT_V2 if schema_version == "v2" else SCHEDULER_SYSTEM_PROMPT_V1
    )

    def _transform(row: dict) -> dict:
        persona = row.get(persona_col, "직장인")
        if not isinstance(persona, str) or not persona.strip():
            persona = "직장인"
        system_content = render_system_prompt(system_tmpl, persona)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(row["prompt"])},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
        }

    return ds.map(_transform, desc="system prompt 적용")


def load_dpo_dataset(
    name: str = "maywell/ko_Ultrafeedback_binarized",
    split: str = "train",
    tokenizer: PreTrainedTokenizer | None = None,
    max_samples: int | None = None,
    max_prompt_len: int = 1024,
    max_response_len: int | None = None,
    min_rejected_len: int = 10,
) -> Dataset:
    # 로컬 parquet/jsonl 파일이면 직접 로드 (HF 캐시 권한 문제 우회)
    if name.endswith(".parquet") and Path(name).exists():
        import pandas as pd
        ds = Dataset.from_pandas(pd.read_parquet(name), preserve_index=False)
    elif Path(name).suffix in (".jsonl", ".json") or Path(name).exists():
        ds = load_dataset("json", data_files=name, split="train")
    else:
        ds = load_dataset(name, split=split)

    missing = _REQUIRED_COLS - set(ds.column_names)
    if missing:
        raise KeyError(
            f"데이터셋 '{name}'에 필수 컬럼 {missing} 없음. "
            f"실제 컬럼: {ds.column_names}. "
            "data/loader.py에 어댑터 함수를 추가하세요."
        )

    # 빈/짧은 rejected 제거 (품질 보장)
    before = len(ds)
    ds = ds.filter(
        lambda r: len(r["rejected"].strip()) >= min_rejected_len,
        desc="rejected 최소 길이 필터",
    )
    removed = before - len(ds)
    if removed:
        print(f"[data] rejected<{min_rejected_len}자 제거: {removed}개 ({before}→{len(ds)})")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    if tokenizer is not None:
        _max_resp = max_response_len

        def _within_len(row: dict) -> bool:
            prompt_ids = tokenizer(row["prompt"], truncation=False)["input_ids"]
            if len(prompt_ids) > max_prompt_len:
                return False
            if _max_resp is not None:
                if len(tokenizer(row["chosen"], truncation=False)["input_ids"]) > _max_resp:
                    return False
                if len(tokenizer(row["rejected"], truncation=False)["input_ids"]) > _max_resp:
                    return False
            return True

        before = len(ds)
        ds = ds.filter(_within_len, desc="시퀀스 길이 필터")
        print(f"[data] 길이 필터 후: {len(ds)}/{before}개 유지 "
              f"(prompt≤{max_prompt_len}, response≤{_max_resp})")

    return ds
