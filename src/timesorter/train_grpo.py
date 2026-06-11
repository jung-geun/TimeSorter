"""GRPO (RLVR) — 시나리오 골격 규칙을 검증 가능한 보상으로 사용하는 online RL.

DPO와의 차이:
  - DPO는 사전 구성된 (chosen, rejected) 쌍의 off-policy 학습 — negative가 프로그램 생성이라
    모델이 실제로 저지르는 오류 분포와 다를 수 있다.
  - GRPO는 학습 중 모델이 직접 생성한 k개 샘플을 골격 규칙(verify_chosen)으로 채점해
    그룹 내 상대 우위를 보상으로 쓴다 — 모델의 진짜 실수가 곧바로 벌점되는 on-policy 학습이며,
    보상 함수가 결정적·무비용(API $0)이라 RLVR 조건을 충족한다.

데이터셋: meta(골격 JSON) 컬럼이 있는 parquet (scheduler_v3/v4). chosen은 사용하지 않는다.

사용:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m timesorter.train_grpo --config configs/grpo_rtx12g_4b_v4.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from .config import RunConfig
from .data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V3,
    parse_lenient,
    render_system_prompt,
)
from .device import detect
from .model import load_model_and_tokenizer


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


# scripts/의 골격 검증기를 재사용 (verify_chosen은 데이터 생성·평가·보상의 단일 출처)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def make_reward_fn():
    from gen_schedule_v3 import Skeleton, TaskSpec, verify_chosen

    def reward_fn(completions: list[str], meta: list[str] | None = None, **kwargs) -> list[float]:
        """보상: 파싱 실패 -1.0 / 규칙 위반당 -0.3 (하한 -1.0) / 전 규칙 통과 +1.0."""
        rewards: list[float] = []
        for comp, m in zip(completions, meta or []):
            try:
                text = comp if isinstance(comp, str) else str(comp)
                parsed = parse_lenient(text)
                if parsed is None or not parsed.tasks:
                    rewards.append(-1.0)
                    continue
                md = json.loads(m)
                skel = Skeleton(scenario=md["scenario"], today=md["today"],
                                specs=[TaskSpec(**s) for s in md["specs"]])
                errors = verify_chosen(skel, parsed.model_dump_json())
                rewards.append(1.0 if not errors else max(-1.0, 1.0 - 0.3 * len(errors) - 0.3))
            except Exception:
                # 보상 함수 예외가 학습 전체를 죽이면 안 됨 — 형식 불량으로 간주
                rewards.append(-1.0)
        return rewards

    reward_fn.__name__ = "skeleton_rule_reward"
    return reward_fn


def build_dataset(parquet_path: str, max_samples: int | None, tokenizer):
    import pandas as pd
    from datasets import Dataset

    df = pd.read_parquet(parquet_path)
    if "meta" not in df.columns:
        raise KeyError(f"{parquet_path}에 meta(골격) 컬럼이 필요합니다.")
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=46).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        system = render_system_prompt(
            SCHEDULER_SYSTEM_PROMPT_V3, str(r.get("persona", "직장인")),
            today=str(r.get("today", "") or ""))
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": str(r["prompt"])}],
            tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "meta": str(r["meta"])})
    return Dataset.from_list(rows)


def main(config_path: str) -> None:
    _load_dotenv()
    cfg = RunConfig.from_yaml(config_path)
    profile = detect()
    print(f"[device] {profile.device} | dtype={profile.dtype}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model_name,
        profile=profile,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        use_4bit=cfg.lora.use_4bit,
        gradient_checkpointing=cfg.training_args.get("gradient_checkpointing", False),
        sft_adapter_path=cfg.sft_adapter,
    )

    ds = build_dataset(cfg.dataset, cfg.max_samples, tokenizer)
    print(f"[data] GRPO 프롬프트 {len(ds)}개")

    training_kwargs: dict = {
        "output_dir": cfg.output_dir,
        "bf16": profile.device == "cuda",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    training_kwargs.update(cfg.training_args)

    if training_kwargs.get("report_to") == "wandb":
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**training_kwargs),
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn()],
    )
    trainer.train()

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    print(f"[done] GRPO 어댑터 저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
