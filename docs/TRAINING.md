# 학습 가이드 (설정 · 하드웨어 · 모듈)

## 모델 구성

| 항목 | 값 |
|------|----|
| 베이스 모델 | **Qwen/Qwen3.5-4B** (기본), **Qwen/Qwen3.5-9B** (24GB+) |
| 어댑터 | LoRA (r=16, alpha=32) |
| 학습 단계 | Stage 1: SFT → Stage 2: DPO (→ GRPO 선택) |
| DPO trick | `ref_model=None` PEFT 트릭으로 메모리 절감 |
| 이력 주의 | `outputs/*` v1~v5 어댑터 중 `*_q35_*`만 Qwen3.5 기반. 나머지는 Qwen3-4B-Instruct-2507 구세대. 어댑터별 베이스는 `adapter_config.json`으로 확인 |

## VRAM 자동 조정 (auto_batch)

실행 시점 VRAM·GPU 수·모델 크기를 감지해 배치·grad_accum·4bit를 자동 산출.

| VRAM | 모델 | bs/GPU | grad_accum | 4bit | eff_batch |
|------|------|--------|-----------|------|-----------|
| 12 GB | 4B | 1 | 16 | ✓ | 16 |
| 24 GB | 4B | 4 | 4 | ✗ | 32 |
| 24 GB | 9B | 1 | 32 | ✓ | 32 |
| 80 GB | 4B | 8 | 4 | ✗ | 32 |

## 스키마 버전

| 버전 | 출력 | 용도 |
|------|------|------|
| v1 | 번호+이름+이유 자유 텍스트 | 기본 정렬 |
| v2/v3 | 4축 점수 JSON (+today/meta) | 구조화·앱 연동·자동 채점 |

---

## 학습 실행 (각 단계 단일 파일)

```bash
# SFT (기본: configs/sft_rtx12g_q35_4b_v4.yaml — Qwen3.5-4B, curated + 프롬프트 loss 마스킹)
uv run python scripts/train_sft.py
uv run python scripts/train_sft.py --config configs/sft_4090_4b_v4.yaml   # 24GB bf16

# DPO (기본: configs/dpo_rtx12g_q35_4b_v5.yaml — on-policy + hard tier)
uv run python scripts/train_dpo.py

# GRPO (configs/grpo_rtx12g_4b_v4.yaml — verify_chosen 보상 RLVR)
uv run python scripts/train_grpo.py
```

- 12GB GPU: 학습 전 서빙 중지 필수 — `docker stop timesorter-serve`
- DPO/GRPO 메모리(12GB): `max_length 1536`, `precompute_ref_log_probs`
- 9B(24GB): SFT 11.5h + DPO 2.5h (4-bit QLoRA). 설정·결과 [experiments/4090_1x_9b_v4/README.md](../experiments/4090_1x_9b_v4/README.md)

### 파이프라인 (Makefile)

```bash
make pipeline-auto       # VRAM 자동 감지 (v1)
make pipeline-auto-v2    # v2 JSON
make pipeline-4090-2x-4b # RTX 4090 × 2
make pipeline-docker     # RTX 12GB Docker
make pipeline-dgx-4b     # DGX 4B
```

### 학습 로깅 (wandb)

`WANDB_API_KEY` 있으면 원격, 없으면 자동 오프라인(`wandb/offline-run-*/`). 나중에: `wandb sync wandb/offline-run-<...>`

---

## 검증 (학습된 어댑터 → 한 줄)

```bash
bash scripts/validate_model.sh outputs/sft_q35_4b_v4              # 30건 빠른 채점
FULL=1 bash scripts/validate_model.sh outputs/dpo_q35_4b_v5       # 150건 + guard rerank
MODE=local bash scripts/validate_model.sh outputs/sft_q35_4b_v4   # GPU 서버 없이 단건
```

4-way 벤치마크(base/no-sft/SFT/DPO, schema+content):
```bash
uv run python scripts/benchmark_q35_4way.py --target all --limit 150   # 4B
uv run python scripts/benchmark_9b.py --target all --limit 150         # 9B (24GB)
```

판사(gpt-5.5) 검증은 `scripts/validate_schedule.py` 참고.

---

## 모듈 구조

```
src/timesorter/
├── device.py        — VRAM 감지 + auto_batch_config
├── config.py        — YAML → RunConfig
├── model.py         — Qwen3.5 로딩 + LoRA
├── data/
│   ├── loader.py    — parquet → DPO 포맷
│   ├── scheduler.py — SFT 데이터 → ChatML
│   └── schema.py    — JSON 스키마 + parse_or_repair
├── train_sft.py / train_dpo.py / train_grpo.py
└── infer.py         — 어댑터 로드 + 생성

scripts/
├── gen_schedule_v3.py        — 골격 우선 데이터 생성 + verify_chosen
├── build_versioned_datasets.py / upload_hf_versioned.py — HF 버전 업로드
├── benchmark_q35_4way.py / benchmark_9b.py — 4-way 벤치마크
└── build_readme_charts.py    — 결과 차트 재생성
```
