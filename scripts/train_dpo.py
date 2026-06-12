#!/usr/bin/env python
"""DPO 학습 단일 실행 파일.

사용:
  uv run python scripts/train_dpo.py                       # 최신 권장 설정 (v5 on-policy+hard)
  uv run python scripts/train_dpo.py --config configs/dpo_rtx12g_4b_v3.yaml

데이터: data/dpo_pairs_v5.parquet (HF: config=dpo — tier=='hard' + refusal 혼합 권장,
        legacy_text(v1)·easy_format(v2 대량)은 docs/DATASET_AUDIT.md 참고)
12GB GPU 메모리 주의: max_length 1536 + precompute_ref_log_probs 필수 (configs 참고).
WANDB_API_KEY가 없으면 자동으로 로컬 오프라인 기록(wandb/)으로 전환된다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.train_dpo import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeSorter DPO 학습")
    parser.add_argument("--config", default="configs/dpo_rtx12g_q35_4b_v5.yaml",
                        help="YAML 설정 (기본: RTX 12GB v5 on-policy)")
    args = parser.parse_args()
    main(args.config)
