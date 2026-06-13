# TimeSorter 발표자료

한국어 일정 우선순위 정렬 모델(Qwen3.5-4B) 파인튜닝 발표용 정리.

## 구성

| 파트 | 내용 | 파일 |
|------|------|------|
| **Part 1** | 4모델(base/instruct/SFT/DPO) 동일 쿼리 출력 비교 + Opus 오류해설 + 통계 | [01_model_comparison/](01_model_comparison/) |
| **Part 2** | 출력 포맷(4축 JSON) 규격화 의도 + SFT/DPO 데이터셋 개형·선호쌍 예시 | [02_output_format/format_spec.md](02_output_format/format_spec.md) |
| **Part 3** | v1→v5 데이터셋 진화 + HF 원본(anakin87·Nemotron) 증강 설명 | [03_dataset_evolution/evolution.md](03_dataset_evolution/evolution.md) |

## Part 1 상세 파일

- [`analysis.md`](01_model_comparison/analysis.md) — 6쿼리 × 4모델 출력 분석, 모델별 오류해설
- [`content_analysis.md`](01_model_comparison/content_analysis.md) — **포맷(스키마) vs 추론(내용) 분리 채점** (no-FT 모델의 숨은 추론 능력)
- [`stats.md`](01_model_comparison/stats.md) — 정량 정확도(n=30) + 6쿼리 통과 요약
- [`raw_outputs.json`](01_model_comparison/raw_outputs.json) — 4모델 전체 raw 출력 + 채점 결과 (원자료)
- [`content_n30.json`](01_model_comparison/content_n30.json) — n=30 스키마/내용 분리 채점 결과

## 핵심 메시지 (발표 3줄 요약)

1. **파인튜닝의 1차 효과는 스키마 준수** — instruct 모델은 스키마 0%지만 **내용만 채점하면 43.3%** (포맷만 못 맞춤). SFT가 규격을 고정해 90%로. 동시에 날짜·체인 같은 진짜 약점엔 실제 추론도 보강.
2. **성능 점프는 데이터 품질·loss 마스킹에서** — DPO/GRPO 선호학습 단독으로는 한계 약점을 못 옮김.
3. **의존성 체인은 미해결 과제** — SFT·DPO가 글자 단위로 동일하게 실패. 체인은 *능력 부재*라 선호학습이 아닌 SFT 데이터 증량이 필요.

## 재현

```bash
# Part 1 추론 (4모델 × 6쿼리, GPU 필요)
uv run python scripts/presentation_inference.py
# 통계 생성
uv run python scripts/presentation_stats.py
# 벤치마크 차트 (README assets용)
uv run python scripts/build_readme_charts.py
```

생성일: 2026-06-13
