# TimeSorter v9 — 모델 검증 결과 비교표

**모델**: Qwen/Qwen3.5-4B + LoRA (r=16, alpha=32)  
**평가셋**: `data/scheduler_v9_eval.parquet`  
**검증기**: `verify_chosen_v9` (8-규칙 결정론적 검증)

---

## v9p1 (EN 증강 학습셋) — 평가셋 n=50

> 평가셋: `data/scheduler_v9_eval.parquet` (KR 31 + EN 28 = 59행, 9행 입력 과다로 제외)  
> 평가 일자: 2026-06-23

| 모델 | parse_rate | verify_pass | chain_order | rank_exact | axis_mae | total_mae |
|------|-----------|-------------|-------------|------------|----------|-----------|
| Base (Qwen3.5-4B) | 88.0% | 4.5% | 34.1% | 0.363 | 2.949 | 2.134 |
| **SFT v9combined** | **96.0%** | **66.7%** | **77.1%** | **0.682** | **0.800** | **0.826** |
| DPO v9combined | 98.0% | 65.3% | 75.5% | 0.667 | 0.823 | 0.849 |
| SFT v9p1 (2,882행) | 88.0% | 45.5% | 47.7% | 0.612 | 0.585 | 0.570 |

### v9p1 주요 관찰 (퇴행 분석)
- **axis_mae/total_mae 개선**: 0.585/0.570 (v9combined 0.800/0.826 대비 -27%)
- **구조 지표 퇴행**: verify_pass -21pp, chain_order -29pp, parse_rate -8pp
- **parse_rate 88% 회귀**: base 수준으로 하락 — EN4 데이터의 JSON 포맷 일관성 문제 추정
- **원인 가설**:
  1. EN4(889행) 데이터 품질: `verify_chosen_v9` 통과했으나 chain/reasoning 품질 저하
  2. 학습셋 비율 변화: KR 47% → EN 53%, EN4 chaining_detail 희박 행 다수
  3. 학습 손실 이상: train_loss 3.256 (v9combined 대비 높음) — EN4 응답 다양성 증가로 학습 수렴 어려움
- **다음 조치**: EN4 데이터 품질 감사 → 저품질 행 제거 후 v9p1 재학습 또는 v9combined로 DPO 진행

---

## v9 종합 학습셋 (v9combined) — 평가셋 n=50 (59행 중 입력 ≤1900 토큰 통과)

> 평가셋: `data/scheduler_v9_eval.parquet` (KR 31 + EN 28 = 59행, 9행 입력 과다로 제외)  
> 평가 일자: 2026-06-23

| 모델 | parse_rate | verify_pass | chain_order | rank_exact | axis_mae | total_mae |
|------|-----------|-------------|-------------|------------|----------|-----------|
| Base (Qwen3.5-4B) | 88.0% | 4.5% | 34.1% | 0.363 | 2.949 | 2.134 |
| **SFT v9combined** | **96.0%** | **66.7%** | **77.1%** | **0.682** | **0.800** | **0.826** |
| DPO v9combined | 98.0% | 65.3% | 75.5% | 0.667 | 0.823 | 0.849 |

### v9combined 주요 관찰
- SFT v9combined: 구 v9 SFT(46.7%) 대비 verify_pass **+20.0pp**, chain_order **+23.8pp** 향상
- DPO가 SFT보다 소폭 낮음 (parse_rate 제외) — chain_order_break/rank_score_mismatch 집중 DPO의 한계
- 동일 패턴: v9 실험에서도 DPO ≤ SFT (45.7% vs 46.7%)
- 원인 추정: overdue+urgent 태스크(R8) 및 EN 행이 DPO 학습 데이터에 미반영

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
