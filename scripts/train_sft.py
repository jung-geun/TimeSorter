#!/usr/bin/env python
"""SFT 학습 단일 실행 파일.

사용:
  uv run python scripts/train_sft.py                       # 최신 권장 설정 (v4 curated)
  uv run python scripts/train_sft.py --config configs/sft_rtx12g_4b_v3.yaml

데이터: data/sft_v4_train.parquet (HF: pieroot/timesorter-scheduler-ko, config=sft, tier 필터)
WANDB_API_KEY가 없으면 자동으로 로컬 오프라인 기록(wandb/)으로 전환된다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.train_sft import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeSorter SFT 학습")
    parser.add_argument("--config", default="configs/sft_rtx12g_4b_v4.yaml",
                        help="YAML 설정 (기본: RTX 12GB v4 curated)")
    args = parser.parse_args()
    main(args.config)
