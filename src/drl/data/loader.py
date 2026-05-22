from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

_REQUIRED_COLS = {"prompt", "chosen", "rejected"}


def load_dpo_dataset(
    name: str = "maywell/ko_Ultrafeedback_binarized",
    split: str = "train",
    tokenizer: PreTrainedTokenizer | None = None,
    max_samples: int | None = None,
    max_prompt_len: int = 1024,
    max_response_len: int | None = None,
    min_rejected_len: int = 10,
) -> Dataset:
    # 로컬 parquet/jsonl 파일이면 직접 로드, 아니면 HF Hub에서 로드
    if Path(name).suffix in (".parquet", ".jsonl", ".json") or Path(name).exists():
        ds = Dataset.from_parquet(name) if name.endswith(".parquet") else load_dataset(
            "json", data_files=name, split="train"
        )
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
