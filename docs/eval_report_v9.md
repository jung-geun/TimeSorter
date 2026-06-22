# TimeSorter v9 — 모델 검증 결과 비교표

**모델**: Qwen/Qwen3.5-4B + LoRA (r=16, alpha=32)  
**평가셋**: `data/scheduler_v9_eval.parquet`  
**검증기**: `verify_chosen_v9` (8-규칙 결정론적 검증)

---

## v9 종합 학습셋 (v9combined) — 평가셋 59행 (KR 31 + EN 28)

| 모델 | parse_rate | verify_pass | chain_order | rank_exact | axis_mae | total_mae |
|------|-----------|-------------|-------------|------------|----------|-----------|
| Base (Qwen3.5-4B) | - | - | - | - | - | - |
| SFT v9combined | - | - | - | - | - | - |
| DPO v9combined | - | - | - | - | - | - |

> 결과 업데이트 예정 (eval 진행 중)

---

## 이전 실험 결과 (v9 — 평가셋 50행 기준)

| 모델 | parse_rate | verify_pass | chain_order | rank_exact | axis_mae | total_mae |
|------|-----------|-------------|-------------|------------|----------|-----------|
| Base (Qwen3.5-4B) | 82.0% | 9.8% | 29.3% | 0.404 | 2.703 | 1.950 |
| SFT v9 (KR 1358+EN 635) | - | 46.7% | 53.3% | 0.591 | 0.671 | - |
| DPO v9 (chain+rank focus) | - | 45.7% | 52.2% | 0.589 | 0.743 | - |

---

## 학습 설정

| 항목 | SFT v9combined | DPO v9combined |
|------|----------------|----------------|
| 데이터 | `scheduler_v9_combined.parquet` (1,993행) | `dpo_pairs_v9_focus.parquet` (2,812쌍) |
| 에폭 | 2 | 1 |
| max_seq_length | 3,584 | 3,584 |
| LoRA r / alpha | 16 / 32 | 16 / 32 |
| lr | 2.0e-5 | 5.0e-7 |
| DPO beta | - | 0.1 |
| DPO 카테고리 | - | chain_order_break + rank_score_mismatch |

---

## 평가 메트릭 정의

| 메트릭 | 설명 |
|--------|------|
| `parse_rate` | 출력이 v9 JSON 스키마로 파싱 성공한 비율 |
| `verify_pass` | `verify_chosen_v9` 8규칙 모두 통과 비율 |
| `chain_order` | 체인 선행→후행 순서 준수 비율 |
| `rank_exact` | gold priority_rank와 정확 일치 비율 |
| `axis_mae` | 4축 점수 평균 절대 오차 (vs gold, 0-10 스케일) |
| `total_mae` | total_score 평균 절대 오차 |

---

## 데이터셋 약점 분석 (v9 → v9.1 증강 근거)

| 약점 | 현황 | 목표 |
|------|------|------|
| EN 데이터 비중 | 31.9% (635행) | 50%+ (1,300행+) |
| is_overdue 비율 | ~3.6% | 15%+ |
| risk_clause 비율 | ~6.8% | 15%+ |
| 고복잡도(10-13 태스크) | ~12% | 20%+ |

---

*생성일: 2026-06-23*
