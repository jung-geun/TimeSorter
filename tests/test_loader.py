import pytest
from unittest.mock import patch

from datasets import Dataset


def _make_ds(**extra):
    base = {
        "prompt": ["한국어 질문입니다?"] * 5,
        "chosen": ["좋은 답변입니다."] * 5,
        "rejected": ["이것은 품질이 낮은 답변입니다."] * 5,
    }
    base.update(extra)
    return Dataset.from_dict(base)


def test_columns_present():
    mock_ds = _make_ds()
    with patch("drl.data.loader.load_dataset", return_value=mock_ds):
        from drl.data.loader import load_dpo_dataset
        ds = load_dpo_dataset(max_samples=3)
    assert {"prompt", "chosen", "rejected"} <= set(ds.column_names)
    assert len(ds) == 3


def test_max_samples_respected():
    mock_ds = _make_ds()
    with patch("drl.data.loader.load_dataset", return_value=mock_ds):
        from drl.data.loader import load_dpo_dataset
        ds = load_dpo_dataset(max_samples=2)
    assert len(ds) == 2


def test_missing_column_raises():
    bad_ds = Dataset.from_dict({"prompt": ["q"], "chosen": ["a"]})
    with patch("drl.data.loader.load_dataset", return_value=bad_ds):
        from drl.data.loader import load_dpo_dataset
        with pytest.raises(KeyError, match="rejected"):
            load_dpo_dataset()
