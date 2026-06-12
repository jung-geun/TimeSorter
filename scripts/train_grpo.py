#!/usr/bin/env python
"""GRPO(RLVR) 학습 단일 실행 파일 — 골격 규칙(verify_chosen)을 보상으로 사용.

사용:
  uv run python scripts/train_grpo.py                      # 파일럿 설정 (12GB)
  uv run python scripts/train_grpo.py --config configs/grpo_rtx12g_4b_v4.yaml

데이터: data/grpo_prompts_v4.parquet (HF: config=grpo — meta 골격 컬럼 필수)
주의: HF generate rollout은 ~9분/스텝 (12GB 기준) — 본 학습은 vLLM rollout 또는 DGX 권장.
WANDB_API_KEY가 없으면 자동으로 로컬 오프라인 기록(wandb/)으로 전환된다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.train_grpo import main



def _stop_serving_container() -> None:
    """학습 전 검증용 vLLM 컨테이너(timesorter-serve)를 자동 중지 — GPU 단독 점유 규칙."""
    import subprocess
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=10).stdout
        if "timesorter-serve" in out:
            print("[GPU] 검증 컨테이너 timesorter-serve 중지 후 학습 시작")
            subprocess.run(["docker", "stop", "timesorter-serve"], timeout=60)
            import time
            time.sleep(3)
    except Exception:
        pass  # docker 미설치 환경(Mac/DGX 베어메탈)은 무시


_stop_serving_container()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeSorter GRPO 학습")
    parser.add_argument("--config", default="configs/grpo_rtx12g_4b_v4.yaml",
                        help="YAML 설정 (기본: RTX 12GB 파일럿)")
    args = parser.parse_args()
    main(args.config)
