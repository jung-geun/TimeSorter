# 데이터셋 버저닝 규칙

TimeSorter 학습 데이터는 **SFT용·DPO용을 분리**하고, **버전별 증분(incremental)** 으로 관리한다.

## 원칙

1. **SFT / DPO 분리** — 두 단계의 데이터는 형식·용도가 달라 별도 HF repo로 관리한다.
   - SFT: [pieroot/timesorter-sft-ko](https://huggingface.co/datasets/pieroot/timesorter-sft-ko)
   - DPO: [pieroot/timesorter-dpo-ko](https://huggingface.co/datasets/pieroot/timesorter-dpo-ko)
2. **증분 단위 버저닝** — 각 `vN`은 그 버전에서 **새로 생성된 데이터만** 담는다(누적 아님).
   무엇이 언제·왜 추가됐는지 추적 가능. origin source 태그로 분리해 재현성 보장.
3. **학습 시 누적 사용** — `vN` 모델 학습 = `v1`..`vN` 증분을 누적해서 사용한다.
   (`scripts/build_hf_release.py`가 누적 학습본을 생성)
4. **앞으로 새 버전 추가 절차**:
   ```
   1. 신규 생성 행만 모아 data/scheduler_vN_*.parquet / dpo_pairs_vN_*.parquet 로 기록
   2. scripts/build_versioned_datasets.py 의 SFT_SPEC/DPO_SPEC 에 vN 항목 추가
   3. uv run python scripts/build_versioned_datasets.py   # 증분 파일 생성
   4. uv run python scripts/upload_hf_versioned.py        # 2개 repo에 vN config 추가
   ```

## v6 — 통합·검수 자립형 데이터셋 (권장)

`v1`~`v5`가 증분이라면, **`v6`은 v2~v5 전체를 검수·통합한 자립형 단일 데이터셋**이다.
증분 누적 없이 **`v6` 단독으로 학습**할 수 있다 (HF 기본 config).

| 구분 | 행 수 | 구성 |
|------|------|------|
| SFT v6 | **14,314** | 검수된 v2 10,958 + 큐레이션 v3-v5 3,356 (drop 0) |
| DPO v6 | **17,894** | 검수된 v2 15,321 + 큐레이션 v3/v5 2,573 (drop 0) |

**검수 방법** (`scripts/v6_prep.py` → workflow → `scripts/v6_assemble.py`):
- v3-v5(meta 보유): `verify_chosen` 골격 규칙 자동검증 통과분만 포함
- v2(meta 없음): **opus 하위 에이전트 264개**로 전수 검수·수정 (100행/배치)
  - SFT: priority_order↔4축 점수 정합 1,188건 수정, JSON·스키마 검증
  - DPO: chosen 정답성·rejected valid-but-worse 4,732건 수정
- **최대한 보존**: drop 0. DPO rejected 중 2,821건은 malformed JSON 유지 — "잘못된 출력 회피"를 가르치는 정당한 negative.

**향후**: v6 이후 새 데이터는 v7 증분으로 추가하거나, 다시 전체 검수해 v7 자립형으로 통합.

## v7 — 의존성 체인 특화 증분 (SFT-only)

v6의 유일한 약점인 의존성 체인을 타깃한 **SFT 증분**. DPO는 v6 그대로(체인은 DPO로 못 옮김).

| 구분 | 행 수 | 구성 |
|------|------|------|
| SFT v7 증분 | **968** | `dependency_chain_complex` (페르소나별 4-5단계·다중 체인) |
| SFT v7 자립형 | **15,282** | v6 14,314 + 체인 968 (`hf_versioned/sft/v7_selfcontained.parquet`) |
| DPO v7 | 17,894 | = v6 무변경 |
| 체인 held-out | 50 | `scheduler_v7_chain_eval.parquet` (seed 777, 학습 미사용) |

**2단계 검수**: ① `verify_chosen` 결정론 검증(라벨↔골격) → ② **opus 블라인드 검수**(텍스트만으로 체인 복원 → 골격 대조, 50행/배치). 1,400 생성 → 968 수록(~70%). 생성은 Sonnet 4.6 위주(Haiku는 복잡 골격에서 품질 미달로 대부분 재생성). 상세: [docs/DATASETS.md](docs/DATASETS.md#v7).

## SFT 버전별 증분

| 버전 | 행 수 | 제목 | 증분 목적 |
|------|------|------|----------|
| `v1` | 5,999 | 자유 텍스트 우선순위 | events-scheduling(영문)→한국어 할 일 번역 + Nemotron 페르소나 주입. '1) 할일 - 이유' 자유 텍스트 출력의 기본 셋. |
| `v2` | 10,958 | 4축 점수 JSON | v1 내용을 4축(긴급/중요/의존/시간) 점수 JSON으로 재생성. 구조화·앱 연동·설명가능성 확보. refusal·orca·xlam 혼합 포함. |
| `v3` | 1,978 | 골격 기반 + 오늘 날짜 주입 | 시나리오 골격(skeleton) 우선 생성 + 오늘 날짜·지난 일정 규칙 추가. 골격 검증(verify_chosen) 통과분만 수록, meta 보존. |
| `v4` | 989 | 이중 생성 경로 + 신규 시나리오 3종 | OpenAI(gpt-5.4)·Claude(Sonnet 4.6/Opus 4.8) 이중 생성으로 표현 다양화. past_split·no_today·am_escalation 신규 시나리오 추가. |
| `v5` | 389 | 의존성 체인 보강 | 잔여 약점(체인)을 타깃해 Claude가 4-5단계 긴 의존성 체인 시나리오 증량 생성. |

## DPO 버전별 증분

| 버전 | 행 수 | 제목 | 증분 목적 |
|------|------|------|----------|
| `v1` | 1,469 | 초기 선호쌍 | v1 우선순위 출력의 선호/비선호 쌍. |
| `v2` | 15,338 | 4축 JSON 선호쌍 + refusal | v2 JSON 포맷 선호쌍 + 비일정 입력 거절(refusal) 쌍 대량 추가. |
| `v3` | 2,222 | hard negative 5종 | 형식은 chosen과 동일하고 내용 한 곳만 틀린 hard negative 도입 (date_confusion/granularity_swap/dependency_scatter/risk_ignore/order_score_mismatch). v2_replay는 v2 재사용이므로 증분에서 제외. |
| `v5` | 351 | on-policy 선호쌍 | SFT v4 모델이 실제로 생성한 오답을 rejected로 수집(on-policy). 체인·dated 오류 집중. |

> **v4 DPO 없음**: v4 선호 단계는 GRPO 파일럿이었고 동률(57.3%)에 그쳐 별도 DPO 증분으로 포함하지 않는다. v4는 SFT 증분만 존재.

## 학습 단계 ↔ 버전 매핑 (실제 이력)

| 모델 버전 | SFT 학습 데이터 | 선호/RL 데이터 | 비고 |
|----------|----------------|---------------|------|
| v1 | sft v1 | dpo v1 | 자유 텍스트 |
| v2 | sft v1~v2 누적 | dpo v1~v2 | 4축 JSON 전환 |
| v3 | sft v1~v3 누적 | dpo v1~v3 | 날짜 추론 |
| v4 | sft v1~v4 누적(curated) | — (GRPO 파일럿) | +20.6%p 돌파 |
| v5 | sft v1~v4 (재사용) | dpo v5 (on-policy) | Qwen3.5-4B, 90.0% |

## 왜 증분 단위인가

- **추적성**: 각 버전이 *무엇을* 추가했는지 한눈에. 성능 변화를 증분과 연결해 분석 가능.
- **재현성**: literal union 재조합이 아니라 실제 생성분만 보존 → 학습한 적 없는 데이터가 섞이지 않음.
- **재사용**: 증분을 조합해 임의 누적본 구성 가능(`build_hf_release.py`).
