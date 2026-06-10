# 발표 자료(PPT) ↔ 현재 코드베이스 갭 분석 및 v3 개선 방향

> 분석 일자: 2026-06-10
> 대상: 팀 발표 자료 (소석영·김경한·봉정근, "일정·할일 우선순위 자동 정렬 비서") 18장

## 1. PPT가 그리는 아키텍처

```
① 입력(자연어 할일) → ② LLM 파서(few-shot, 태스크별 urgency/importance/category JSON)
→ ③ Feature Generator(가중합: 긴급 0.50 + 중요 0.30 + 마감 0.10 + 정렬보너스 0.10)
→ ④ ScoreRanker(결정적 정렬) → ⑤ 사용자 피드백(순서 수정 → chosen/rejected) → ⑥ DPO 학습
```

- 마감일 점수표: D+ 1.00 / D-0 0.95 / D-1 0.85 / … / 없음 0.00 (오늘 날짜 기준 감쇠)
- 정렬보너스 = urgency_norm × importance_norm (아이젠하워 1사분면 강조)
- PPT 기준 ⑤ 피드백 저장, ⑥ DPO 학습은 **미구현**으로 표기 (베이스도 Mistral-7B로 현재와 다름)

## 2. 현재 코드베이스와의 차이

| 항목 | PPT | 현재 레포 | 판단 |
|------|-----|----------|------|
| 구조 | 파서→피처→랭커 분해 파이프라인 | 단일 LLM이 4축 채점+순서 동시 출력 (end-to-end) | 절충 (아래) |
| 점수→순서 | 가중합으로 **결정적** 도출 | 모델이 priority_order를 별도 생성 → 점수와 모순 가능 | **PPT 방식 채택** |
| 오늘 날짜 | deadline_score가 오늘 기준 D-day 감쇠 | 시스템 프롬프트에 오늘 날짜 없음 → 날짜 혼동 (검증 FAIL 1순위 원인) | **PPT 방식 채택** |
| 태스크별 채점 | 파서가 기본값 3/3을 남발 → 동점 다발 (slide 6: 6개 중 4개 0.425 동점) | 학습된 모델이라 분별력 있음 | 현재 방식 유지 |
| 피드백→DPO | 설계만 존재 (미구현) | 미구현이었음 | **신규 구현** |
| 베이스 모델 | Mistral-7B-Instruct-v0.2 | Qwen3-4B-Instruct-2507 + LoRA | 현재 방식 유지 |

핵심 결론: **end-to-end 채점 모델(현재)의 분별력 + PPT의 결정적 랭킹·날짜 기준·피드백 루프**를
결합하는 것이 양쪽의 약점을 상쇄한다.

## 3. v2 검증 실패 모드 → v3 대응표

gpt-5.5 교차 검증(2026-05-24)에서 SFT/DPO v1·v2 전부 종합 2/5 FAIL.

| 실패 모드 | 원인 | v3 대응 |
|----------|------|---------|
| ① 날짜 혼동 (지난 5/23 일정에 urgency=5, 3위 배치) | 학습 데이터 전체가 "이메일 날짜=오늘" 가정, 프롬프트에 오늘 날짜 부재 | `SCHEDULER_SYSTEM_PROMPT_V3`: 오늘 날짜(+요일) 고정 주입 + 지난 일정 처리 규칙 명문화. `dated_mixed` 시나리오(전체 35%)로 과거 일정 케이스 합성. DPO `date_confusion` hard negative |
| ② 오전/오후 마감 미구분 (PR#847 오전 마감 < Q2보고서 17:00) | 동일일 내 시각 비교 케이스 부재 | urgency 가이드에 "오전=5, 오후=4, 당일중=3" 명시. `intraday` 시나리오 20%. DPO `granularity_swap` |
| ③ 의존성 분산 (작성→업로드→발송을 1·7·8위로 산개) | dependency 축이 학습 신호로 약함 | 의존성 규칙(연쇄 작업 연속 배치) 프롬프트 명문화. `dependency_chain` 시나리오 20%. DPO `dependency_scatter` |
| (보조) 에스컬레이션·위약금 무시 | 리스크 신호어 학습 부재 | importance 가이드에 리스크 조항 명시. `risk` 시나리오 15%. DPO `risk_ignore` |
| (보조) 점수↔순서 모순 | 모델이 둘을 독립 생성 | `timesorter/rank.py` ScoreRanker 후처리(`--rerank`) + DPO `order_score_mismatch` |

## 4. v2 DPO의 구조적 문제: 쉬운 negative

DPO v2는 rewards/accuracies=1.0, margins=17.7로 "완벽" 수렴했지만 검증은 FAIL.
rejected 4종(invalid_json·bad_scores·urgency_only·shallow_reason)이 모두 **형식 수준에서
구분되는 쉬운 negative**라서, 모델은 "JSON을 깨지 말 것" 이상을 배우지 못했다.

v3 DPO는 chosen과 **형식·길이가 동일하고 내용만 틀린** hard negative를 시나리오 골격
(어느 태스크가 과거인지·체인인지·리스크인지를 프로그램이 알고 있음)에서 결정적으로 생성한다.
margins가 낮게 유지되더라도 실제 실패 모드를 직접 벌점하는 신호다.

## 5. v3 데이터 품질 장치

1. **골격 우선 생성**: 날짜·체인·리스크 배치는 프로그램이 결정 → GPT는 텍스트만 작성
   (GPT가 날짜를 틀릴 수 없는 구조)
2. **chosen 생성 시 특권 정보**: 생성기에게만 "태스크 N은 이미 지남" 사실 힌트 제공
   (저장되는 prompt에는 미포함 — distillation with privileged information)
3. **프로그래매틱 검증**: 과거 일정 후순위 / 동일일 시각 순서 / 체인 연속성·순서 /
   리스크 importance / 무마감 1위 금지 — 위반 시 위반 내용을 알려주고 1회 재생성, 그래도 실패면 폐기
4. **v2 replay 혼합**: SFT에 refusal 전체 + 일반 2.5K, DPO에 v2 쌍 3K → 기존 능력(거부·형식) 망각 방지

## 6. 학습 계획 (RTX 3080 Ti 12GB)

| 단계 | 베이스 | 데이터 | 비고 |
|------|--------|--------|------|
| SFT v3 | sft_rtx12g_4b_v2에서 **이어 학습** | scheduler_v3_combined (~5.7K) | lr 1e-5, 2 epochs |
| DPO v3 | sft_rtx12g_4b_v3 | dpo_pairs_v3 (~7K, hard negative + v2 replay 3K) | lr 5e-7, 1 epoch |

추론 시 `--schema-version v3 --today YYYY-MM-DD` 필수 (학습·추론 프롬프트 일치).
선택적으로 `--rerank`로 점수 기반 결정적 재정렬.

## 7. 남는 항목 (이후 단계)

- PPT의 마감일 감쇠표를 rank.py에 직접 반영하려면 모델 출력에 마감일 파싱 필드 추가 필요 (스키마 확장)
- 캘린더 충돌 감지, 다중 이메일 N건 컨텍스트 (README 장기 계획과 동일)
- 피드백 루프 실서비스 연결: `timesorter/feedback.py`의 FeedbackRecord를 UI에 연결
