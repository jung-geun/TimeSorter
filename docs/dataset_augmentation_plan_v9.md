# TimeSorter v9 데이터셋 증강 계획

**작성일**: 2026-06-23  
**현재 학습셋**: `data/scheduler_v9_combined.parquet` (1,993행)  
**기반 모델**: Qwen/Qwen3.5-4B  
**SFT 어댑터**: `outputs/sft_q35_4b_v9combined`  
**DPO 어댑터**: `outputs/dpo_q35_4b_v9combined`

---

## 1. 현재 데이터셋 현황

### 1.1 규모 및 언어 분포

| 언어 | 행 수 | 비율 |
|------|-------|------|
| KR   | 1,358 | 68.1% |
| EN   | 635   | 31.9% |
| **합계** | **1,993** | 100% |

### 1.2 페르소나 타입 분포

| 카테고리 | KR | EN |
|----------|----|----|
| office_admin | 132 | 55 |
| healthcare | 129 | 49 |
| sales_retail | 128 | 54 |
| transport_logistics | 125 | 51 |
| management | 123 | 43 |
| manual_production | 112 | 53 |
| jobseeker_senior | 108 | 60 |
| general | 107 | 63 |
| professional_tech | 105 | 50 |
| service_food | 99 | 53 |
| security_safety | 99 | 61 |
| education | 91 | 43 |

### 1.3 시나리오 구조

| 항목 | 값 |
|------|----|
| 평균 태스크 수/행 | 7.2개 |
| 최소-최대 | 4~13개 |
| 체인 포함 행 비율 | 82.5% |
| 평균 chain_pairs 수 | 2.6개/행 |
| KR 스캐폴드 is_overdue 비율 | 3.6% |
| EN 스캐폴드 is_overdue 비율 | 3.8% |
| risk_clause 행 비율 (KR) | 6.8% |

### 1.4 복잡도 분포 (태스크 수별 행 수)

| 태스크 수 | 행 수 | 비율 |
|-----------|-------|------|
| 4~6개 (저복잡도) | 216 | 33% |
| 7~9개 (중복잡도) | 308 | 47% |
| 10~13개 (고복잡도) | 76 | 12% |
| 슬롯 없음(이상) | 75 | - |

---

## 2. 확인된 약점

### 약점 1: EN 데이터 절대량 부족 (우선순위: 최상)
- **현상**: EN 635행 = KR의 47%. 언어 불균형으로 EN 추론 품질 저하
- **증거**: eval_en_sft 결과가 KR보다 낮은 chain_order_rate
- **목표**: EN 1,300행 수준으로 증강 (+650행)

### 약점 2: is_overdue(R7) 시나리오 희소 (우선순위: 상)
- **현상**: 스캐폴드에 overdue 태스크가 있으나(~3.6%) 모델이 R7 규칙("마감이 지났지만 긴급 유지") 학습 부족
- **증거**: eval에서 is_overdue 포함 행의 verify_pass가 낮음
- **목표**: overdue 집중 시나리오 KR/EN 각 150행 생성

### 약점 3: risk_clause 부족 (우선순위: 중)
- **현상**: KR 스캐폴드에서 risk 있는 행 6.8%에 불과
- **목표**: risk_clause 시나리오 KR/EN 각 100행 추가

### 약점 4: 고복잡도(10~13 태스크) 시나리오 부족 (우선순위: 중)
- **현상**: 전체 12%만 10개 이상 태스크. 복잡한 하루 스케줄 학습 부족
- **목표**: 고복잡도 시나리오 +100행

### 약점 5: 체인 없는(독립 태스크만) 시나리오 부족 (우선순위: 하)
- **현상**: chain 있는 행 82.5% — 독립 태스크만 있는 날 학습 부족
- **목표**: no-chain 시나리오 +100행

### 약점 6: DPO 개선 폭 미미 (우선순위: 상)
- **현상**: DPO ≈ SFT (v9 결과: verify_pass 46.7%→45.7%)
- **원인 분석**: chain_order_break 위반 시나리오에서 모델이 이미 correct를 선호하지 못함
- **대안**: harder negative 생성 (rank_inversion + schedule_overlap 조합)

---

## 3. 증강 계획

### Phase 1: EN 균형화 (목표: +650행)
- **스캐폴드**: 기존 `scripts/v9/build_dataset_en.py` 파이프라인 재사용
- **배치**: 배치 50개 × 13행 = 650행
- **페르소나**: management, education, healthcare 중심 (현재 적은 카테고리)
- **검증**: `verify_chosen_v9` → opus 감사 → `data/scheduler_v9_en_v2.parquet`

### Phase 2: is_overdue 집중 시나리오 (목표: +300행)
- **스캐폴드 수정**: `build_dataset.py`에 `overdue_ratio=0.4` 파라미터 추가
  - 행당 30~50% 태스크에 is_overdue=True 강제
  - urgency ≥ 8.0으로 자동 설정
- **프롬프트 강화**: sonnet 프롬프트에 R7 규칙 명시 강화
- **출력**: KR 150행 + EN 150행 → `data/scheduler_v9_overdue.parquet`

### Phase 3: DPO 하드 네거티브 강화 (목표: +500쌍)
- **새 카테고리**: `overdue_rank_drop` — overdue 태스크의 rank를 낮게 설정한 negative
- **기존 카테고리 보강**: `chain_order_break` 전체 배치 재생성
- **출력**: `data/dpo_pairs_v9_v2.parquet`

### Phase 4: 고복잡도 시나리오 (목표: +100행)
- **설정**: `n_tasks=10~13` 고정, `chain_pairs ≥ 3`
- **출력**: `data/scheduler_v9_complex.parquet`

---

## 4. 증강 후 목표 규모

| 버전 | SFT 행 수 | EN 비율 | DPO 쌍 수 | eval 행 수 | 상태 |
|------|-----------|---------|-----------|------------|------|
| v9 (기준) | 1,993 | 31.9% | 7,438 | 59 | ✅ 완료 |
| v9p1 (Phase 1) | **2,882** | **52.9%** | 7,438+2,812 | 59 | ✅ 완료 (2026-06-23) |
| v9.2 (Phase 2-3) | ~3,200 | 53%+ | ~8,500 | 100+ | 🔜 예정 |
| v9.3 (Phase 4) | ~3,300 | 53%+ | ~8,500 | 100+ | 🔜 예정 |

### Phase 1 실적 (2026-06-23)
- **scaffold**: `outputs/v9/build_en3/scaffold/` (63배치, 1,000 scaffold 행)
- **fill**: workflow `wjzmt9psw` — haiku(제목/메모) + sonnet(reasoning) 127 에이전트
- **assemble**: 1,000 후보 → 889행 통과 (verify_fail 48, placeholder 55, no_llm 8)
- **combined**: `scheduler_v9_combined.parquet`(1,993) + `scheduler_v9_en4.parquet`(889) = **2,882행**
- **EN 비율**: 635 → 1,524 (31.9% → **52.9%**) ✅

---

## 5. 생성 파이프라인 전체 기록

### 5.1 스캐폴드 생성 (결정적)
```
scripts/v9/build_dataset.py    → KR 스캐폴드 (outputs/v9/build_kr_v2/scaffold/)
scripts/v9/build_dataset_en.py → EN 스캐폴드 (outputs/v9/build_en_v2/scaffold/)
```
- 입력: `data/v9/personas_*.json`, 시나리오 파라미터
- 출력: `batch_NNN.json` (persona, slots, scoring, schedule, chain_pairs, facts)
- 결정적: seed 고정, 재현 가능

### 5.2 LLM 콘텐츠 생성 (워크플로우)
```
.claude/plans/v9_fill_lean_kr.js  → KR haiku(제목/메모) + sonnet(reasoning)
.claude/plans/v9_fill_lean_en.js  → EN haiku(제목/메모) + sonnet(reasoning)
```
- haiku: 태스크 제목/메모/소스 생성
- sonnet: facts 기반 summary + chaining_detail 작성
- opus(eval 전용): 현실성·논리 일관성 감사

### 5.3 검증 및 조립
```
scripts/v9/assemble_v9.py
  --llm <result.json> --lang ko/en
  --scaffold-dir <dir> --out-sft <parquet>
```
- `verify_chosen_v9()`: rank 순열, 블록 중복, 소요 일치, 마감 실현, total 재계산, 체인 선후, chaining_detail 조건부
- 핵심 버그 수정: `load_llm(path, lang=None)` — KR/EN row_id 충돌 방지

### 5.4 합산 및 학습
```
scripts/v9/combine_v9.py → data/scheduler_v9_combined.parquet
scripts/v9/train_sft_v9.py --config configs/sft_rtx12g_q35_4b_v9combined.yaml
scripts/v9/train_dpo_v9.py --config configs/dpo_rtx12g_q35_4b_v9combined.yaml
```

### 5.5 평가 및 업로드
```
scripts/v9/eval_v9.py --adapter base/sft/dpo --data data/scheduler_v9_eval.parquet
scripts/v9/upload_v9.py  → HF pieroot/timesorter-scheduler-v9-ko
```

---

## 6. 논문 방향 (데이터셋 구축 기록)

### 제목 후보
"TimeSorter-v9: A Rule-Grounded JSON Scheduling Dataset with Deterministic Verification"

### 핵심 기여
1. **결정론적 스캐폴드 + LLM 텍스트화** 파이프라인: 점수·순위·시간블록을 먼저 계산하고, LLM은 텍스트(제목/메모/근거)만 생성 → hallucination 방지
2. **8-규칙 검증기 (`verify_chosen_v9`)**: rank 순열, 블록 중복, 소요 일치, 마감 실현, total 재계산, 체인 선후, chaining_detail 조건부, is_overdue 긴급 유지
3. **이중 언어 (KR/EN)**: 동일 구조, 언어별 페르소나 및 프롬프트 분리
4. **DPO hard negative**: 규칙 1개 위반 샘플 자동 생성 (5종 카테고리)
5. **Held-out eval**: 논리 일관성 통과 + opus 감사 통과한 59행 (KR 31 + EN 28)
