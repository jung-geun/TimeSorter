# TimeSorter — 한국어 할 일 우선순위 정렬 비서

> **Qwen3.5-4B / 9B**를 한국어 일정 관리 태스크에 특화 파인튜닝하는 SFT → DPO 2단계 파이프라인.
> 사용자가 제출한 할 일 목록을 **긴급도·중요도·의존성·시간 제약** 4축으로 채점해 우선순위를 결정합니다.

---

## 프로젝트 목적

스마트폰·PC에서 "오늘 할 일"을 입력하면 AI가 맥락을 이해해 실행 순서를 제안하는 개인 비서 코어 모델을 만드는 것이 목표입니다.

단순 키워드 기반 정렬이 아닌, **페르소나**(직장인·학생·부모 등)와 **4가지 축**을 기반으로 각 태스크를 1–5점으로 채점하고 그 근거를 함께 제시합니다.

```
입력: "임원 보고서 마감(내일), 팀 회의(오후 2시), 점심 약속, 메일 답장 3건"

출력:
1) 임원 보고서 마감  [긴급5·중요5·의존4·시간2] — 내일 마감, 핵심 업무
2) 팀 회의(오후 2시) [긴급4·중요4·의존3·시간4] — 고정 시각, 후속 블로킹
3) 메일 답장 3건     [긴급4·중요3·의존2·시간1] — 긴급하나 고정 시각 없음
4) 점심 약속         [긴급2·중요2·의존1·시간3] — 유연 조정 가능
```

---

## 데이터셋 구성

### v1 — 자유 텍스트 우선순위 응답

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_ko.parquet` | 1,200 | GPT 생성 한국어 스케줄 기본 셋 |
| `scheduler_generic.parquet` | 3,000 | 다양한 일상 태스크 확장 |
| `scheduler_nemotron_r2~r4.parquet` | 각 ~2,500 | Nemotron 페르소나 기반 다양화 3라운드 |
| **`scheduler_ko_combined.parquet`** | **5,999** | v1 SFT 통합 (위 파일 병합) |

**응답 형식 (v1)**:
```
1) 보고서 마감 - 외부 고객 신뢰와 직결된 마감이라 가장 우선합니다.
2) 팀 회의 - 협업에 필수적인 정보 공유 자리입니다.
3) 운동 - 건강을 위해 중요하지만 시간 제약이 낮습니다.
```

### v2 — 4축 점수 JSON 응답

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `scheduler_v2_regen.parquet` | ~6,000 | v1 데이터를 v2 JSON으로 재생성 |
| `scheduler_v2_nemotron_extra.parquet` | ~3,000 | Nemotron 페르소나 v2 추가 |
| **`scheduler_v2_combined.parquet`** | **10,958** | v2 SFT 통합 |
| `dpo_pairs_v2.parquet` | DPO용 | 선호/비선호 응답 쌍 (v2 JSON) |

**응답 형식 (v2)**:
```json
{
  "tasks": [{"id": 1, "text": "보고서 마감"}, {"id": 2, "text": "팀 회의"}],
  "priority_order": [1, 2],
  "scores": [
    {"task_id": 1, "urgency": 5, "importance": 5, "dependency": 4,
     "time_constraint": 2, "reason": "내일 마감, 고객사 핵심 업무"},
    {"task_id": 2, "urgency": 4, "importance": 4, "dependency": 3,
     "time_constraint": 4, "reason": "오후 고정 시각, 후속 작업 입력"}
  ]
}
```

### 데이터 구축 방법

1. **한국어 일정 시드 생성**: GPT-4o로 다양한 페르소나·상황의 할 일 목록 생성
2. **Nemotron 페르소나 다양화**: `nvidia/Nemotron-Personas-Korea` 1.8GB 데이터셋을 활용해 직업·연령·라이프스타일별로 3라운드 재생성
3. **응답 품질 검증**: GPT judge로 우선순위 근거 논리성 검증 후 필터링
4. **v2 JSON 변환**: v1 자유 텍스트 응답을 4축 채점 JSON 포맷으로 재생성 (GPT-4o 활용)
5. **DPO 쌍 생성**: 동일 입력에 대해 고품질/저품질 응답 쌍 자동 생성

---

## 학습 특징

### 모델 구성

| 항목 | 값 |
|------|----|
| 베이스 모델 | Qwen/Qwen3.5-4B (기본), Qwen/Qwen3.5-9B (DGX) |
| 어댑터 | LoRA (r=16, alpha=32) |
| 학습 단계 | Stage 1: SFT → Stage 2: DPO |
| DPO trick | `ref_model=None` PEFT 트릭으로 메모리 절감 |

### VRAM 자동 조정 (auto_batch)

실행 시점 VRAM·GPU 수·모델 크기를 감지해 배치 크기·grad_accum·4bit 여부를 자동 산출합니다.

| VRAM | 모델 | bs/GPU | grad_accum | 4bit | eff_batch |
|------|------|--------|-----------|------|-----------|
| 12 GB | 4B | 1 | 16 | ✓ | 16 |
| 24 GB | 4B | 4 | 4 | ✗ | 32 |
| 24 GB×2 | 4B | 4 | 4 | ✗ | 32 |
| 80 GB | 4B | 8 | 4 | ✗ | 32 |
| 120 GB | 9B | 4 | 8 | ✗ | 32 |

### 스키마 버전

| 버전 | 출력 | 용도 |
|------|------|------|
| v1 | 번호+이름+이유 자유 텍스트 | 기본 우선순위 정렬 |
| v2 | 4축 점수 JSON | 구조화된 근거 제공, 앱 연동 가능 |

---

## 학습 결과 (달성 현황)

### RTX 12GB (Qwen3-4B, QLoRA 4-bit)

현재까지 완료된 학습 실험 전체 목록입니다.

| 단계 | 어댑터 경로 | 데이터셋 | 샘플 수 | train_loss | 비고 |
|------|-----------|---------|--------|-----------|------|
| SFT v1 | `outputs/sft_rtx12g_4b` | scheduler_ko_combined | 5,999 | — | 자유 텍스트 출력 |
| DPO v1 | `outputs/dpo_rtx12g_4b` | dpo_pairs_v1 | ~5K쌍 | — | v1 어댑터 위 선호도 학습 |
| **SFT v2** | **`outputs/sft_rtx12g_4b_v2`** | **scheduler_v2_combined** | **10,958** | **—** | **4축 JSON 출력** |
| **DPO v2** | **`outputs/dpo_rtx12g_4b_v2`** | **dpo_pairs_v2** | **15,280쌍** | **0.0043** | **현재 최신 어댑터** |

#### DPO v2 학습 지표 (956 steps, 2 epoch)

| 지표 | 초반 (step 10) | 중반 (step 478) | 최종 (step 956) |
|------|--------------|---------------|---------------|
| train_loss | ~2.0 | ~0.05 | 0.0043 |
| rewards/accuracies | ~0.75 | ~0.98 | 1.0 |
| rewards/margins | ~5.0 | ~15.0 | 17.7 |
| learning_rate | 2e-5 | ~1e-5 (cosine) | ~0 |

> rewards/accuracies=1.0, margins=17.7로 chosen/rejected 완전 분리 달성.
> wandb 프로젝트: https://wandb.ai/pieroot-pieroot/drl-qwen3

#### Mac MPS 비교 (300샘플, 5epoch — 초기 실험)

| 실험 | train_loss | accuracy |
|------|-----------|---------|
| SFT v1 | 0.977 | 76.5% |
| SFT v2 | 0.415 | 90.0% |

> 전체 epoch별 상세 메트릭: [docs/TRAINING_LOG.md](docs/TRAINING_LOG.md)

---

## 검증 결과 및 현재 문제점

### gpt-5.5 교차 검증 결과 (2026-05-24 기준)

| 어댑터 | 태스크 커버리지 | 우선순위 정확도 | 추론 품질 | 4축 일관성 | 종합 | 판정 |
|--------|--------------|--------------|---------|-----------|------|------|
| SFT v1 | 3/5 | 2/5 | 2/5 | — | 2/5 | ❌ FAIL |
| DPO v1 | 3/5 | 2/5 | 2/5 | — | 2/5 | ❌ FAIL |
| SFT v2 | 4/5 | 2/5 | 2/5 | 2/5 | 2/5 | ❌ FAIL |
| **DPO v2** | **4/5** | **2/5** | **2/5** | **2/5** | **2/5** | **❌ FAIL** |

> 검증 방법: Phase 1에서 gpt-5.5가 원본 이메일을 독립 분석해 기준 답을 생성하고,
> Phase 2에서 모델 출력과 비교 채점. `--today` 파라미터로 현재 날짜를 판사에게 주입.

### 확인된 문제점 3가지

#### 1. 날짜 혼동 — 가장 심각

모델이 이메일 작성 날짜(5/23)와 추론 시점(5/24)을 구분하지 못합니다.
이미 지난 5/23 일정(미팅, 회식)에 `urgency=5, time_constraint=5`를 부여하고 3위에 배치했습니다.

```
이메일 날짜: 2026-05-23 (어제)
추론 날짜:   2026-05-24 (오늘)

모델 순위:  1) Q2보고서(5/24 17:00)  2) PR#847(5/24 오전)  3) ★그린테크 미팅(5/23 14:00 — 이미 지남)
정답 순위:  1) PR#847(오전 마감+에스컬레이션)  2) Q2보고서(17:00)  3) 계약서 검토(5/26)
```

**원인**: 학습 데이터에서 이메일 날짜 = 오늘 날짜로 항상 일치. `today ≠ email_date` 케이스가 전혀 없음.

#### 2. 오전/오후 마감 임박도 미반영

같은 날 마감이더라도 "오전 마감(PR #847)"이 "17:00 마감(Q2 보고서)"보다 더 임박함을 인식하지 못합니다.
또한 PR #847에 "미처리 시 팀장 자동 에스컬레이션"이라는 조건이 명시됐음에도 2위 배치했습니다.

#### 3. 태스크 분리 비일관성

Q2 보고서의 "작성 → 공유 드라이브 업로드 → 참조 메일 발송"은 하나의 완결된 업무 흐름인데
모델이 이를 각각 1위·7위·8위로 분산 배치했습니다. 단계별 의존성(dependency 축)이 학습에 충분히 반영되지 않은 결과입니다.

---

## 개선 계획

### 단기 개선 (v0.7 — 데이터 수정)

#### A. `today ≠ email_date` 학습 케이스 추가

```python
# 현재: 모든 샘플이 이메일 날짜 = 오늘
# 개선: 이메일이 1-7일 전에 작성된 케이스를 30% 비중으로 합성
{
  "system": "오늘은 {today}입니다. 아래 이메일 중 이미 지난 일정을 식별하고...",
  "email_date": "2026-05-23",
  "today": "2026-05-24"   # ← 새로 추가할 필드
}
```

생성 전략: `gen_schedule_v2.py`에 `--days-offset 1-7` 옵션을 추가해 과거 날짜 이메일 케이스를 합성.

#### B. 시간 내 순서(오전 < 오후) 채점 강화

학습 데이터에 동일 날짜 내 세분화된 시간 비교 케이스를 추가합니다.
`urgency` 채점 가이드에 "오전 마감=5, 오후 초반 마감=4, 오후 후반 마감=3" 기준을 명시합니다.

#### C. 태스크 의존성 클러스터링 케이스 추가

"A → B → C 순서가 강제되는 태스크 묶음"이 입력될 때 올바르게 높은 `dependency` 점수를 주고
우선순위도 클러스터 단위로 묶어 배치하는 케이스를 DPO rejected에 추가합니다.

```
chosen:  1) 계약서 작성(dep=5) 2) 법무팀 검토(dep=5) 3) 서명(dep=5)
rejected: 1) 계약서 작성(dep=3) 5) 법무팀 검토(dep=1) 9) 서명(dep=1)
```

#### D. 에스컬레이션·위약금 등 리스크 키워드 인식

"미처리 시 ~", "위약금 ~원", "고객사 클레임" 같은 리스크 신호어에 `importance` 점수를 높이는
케이스를 Nemotron 재생성 시 의도적으로 포함합니다.

### 중기 개선 (v0.8 — 아키텍처)

| 개선 | 방법 | 기대 효과 |
|------|------|----------|
| 오늘 날짜 주입 표준화 | system prompt에 `오늘: {date}` 필드를 고정 위치에 배치, 학습·추론 동일 포맷 적용 | 날짜 혼동 근본 해결 |
| 컨텍스트 윈도우 확장 | max_seq_length 2048 → 4096, 12GB VRAM에서 grad_accum 64로 보완 | 긴 이메일 5건 + JSON 응답 완전 수용 |
| 9B 모델 실험 | DGX 환경에서 Qwen3.5-9B SFT+DPO 실행 | 추론 깊이 향상 |

### 장기 개선 (v1.0)

- **다중 이메일 컨텍스트 학습**: 이메일 1건 입력 → 이메일 N건 입력으로 학습 데이터 구성 변경
- **캘린더 연동**: 기존 일정 정보를 컨텍스트에 추가해 충돌 감지
- **사용자 피드백 루프**: 실제 사용자 수정 내역을 RLHF 신호로 활용

### 미완 / 다음 작업

- [ ] `gen_schedule_v2.py --days-offset` 구현 및 today≠email_date 케이스 2K 생성
- [ ] 오전/오후 세분화 urgency 채점 기준 반영 후 DPO 쌍 재생성
- [ ] DGX 환경에서 9B 모델 학습
- [ ] `v0.5-baseline` git tag 추가 (v1 어댑터 보존 포인트)

---

## 빠른 시작

### 1. 환경 설정

```bash
make setup-mac      # Mac (MPS)
make setup-dgx      # DGX / Linux ARM64 CUDA
make docker-build   # RTX GPU (Docker)
```

`.env` 파일:
```
OPENAI_API_KEY=sk-...   # 데이터 생성 필수
HF_TOKEN=hf_...         # 모델 다운로드
WANDB_API_KEY=...       # 학습 모니터링
HF_HOME=models          # 로컬 모델 캐시 (프로젝트 내 저장)
```

→ 상세: [docs/SETUP.md](docs/SETUP.md)

### 2. 데이터 준비

```bash
make download          # HF 데이터셋 다운로드
make download-models   # Qwen3.5-2B / 4B / 9B 가중치 캐싱
```

→ 상세: [docs/DATASET.md](docs/DATASET.md)

### 3. 학습

```bash
# VRAM 자동 감지 (권장)
make pipeline-auto      # v1 자유 텍스트
make pipeline-auto-v2   # v2 JSON 4축 점수

# 하드웨어 직접 지정
make pipeline-4090-2x-4b    # RTX 4090 × 2
make pipeline-docker         # RTX 12GB Docker
make pipeline-dgx-4b         # DGX 4B
```

→ 상세: [docs/TRAINING.md](docs/TRAINING.md)

### 4. 추론

```bash
# v1 자유 텍스트
make infer ADAPTER=outputs/sft_mac \
  PROMPT="보고서 마감(내일), 팀 회의(오후 2시), 메일 답장 3건"

# v2 JSON 4축 점수
uv run python -m timesorter.infer --adapter outputs/sft_mac_v2 \
  --schema-version v2 --persona "직장인" \
  --prompt "보고서 마감(내일), 팀 회의(오후 2시), 메일 답장 3건"

# vLLM 서빙 (포트 8000)
make serve-docker
```

→ 상세: [docs/SERVING.md](docs/SERVING.md)

### 5. 검증

```bash
make validate   # GPT 판사 교차 검증
```

→ 상세: [docs/VALIDATION.md](docs/VALIDATION.md)

---

## 모듈 구조

```
src/timesorter/
├── device.py        — VRAM 감지 + auto_batch_config
├── config.py        — YAML → RunConfig
├── model.py         — Qwen3.5 로딩 + LoRA / DDP 대응
├── data/
│   ├── loader.py    — HF 데이터셋 / parquet → DPO 포맷
│   ├── scheduler.py — SFT 데이터 → ChatML (v1/v2 분기)
│   ├── augment.py   — LLM 생성 + GPT judge
│   └── schema.py    — v2 JSON 스키마 정의 + parse_or_repair
├── train_sft.py     — SFTTrainer 래퍼
├── train_dpo.py     — DPOTrainer 래퍼
└── infer.py         — 어댑터 로드 + 텍스트 생성

configs/
├── sft_auto.yaml / dpo_auto.yaml           — 하드웨어 무관 (auto_batch, v1)
├── sft_auto_v2.yaml / dpo_auto_v2.yaml     — 하드웨어 무관 (auto_batch, v2)
├── sft_4090_2x_4b.yaml                     — RTX 4090 × 2
├── sft_rtx12g_4b.yaml                      — RTX 12GB QLoRA
├── sft_dgx_4b.yaml / sft_dgx_8b.yaml      — DGX 4B / 9B
└── accelerate_4090_2x.yaml                 — 2-GPU DDP
```

---

## 상세 문서

| 문서 | 내용 |
|------|------|
| [docs/SETUP.md](docs/SETUP.md) | 환경 설정, Docker, API 키 |
| [docs/DATASET.md](docs/DATASET.md) | 데이터셋 명세, 생성 파이프라인 |
| [docs/TRAINING.md](docs/TRAINING.md) | 학습 설정, 하드웨어별 옵션 |
| [docs/TRAINING_LOG.md](docs/TRAINING_LOG.md) | epoch별 loss/accuracy 전체 기록 |
| [docs/SERVING.md](docs/SERVING.md) | vLLM 서빙, 이메일 파이프라인 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 교차 검증 방법 및 분석 |
| [docs/BACKLOG.md](docs/BACKLOG.md) | 개선 계획 |
