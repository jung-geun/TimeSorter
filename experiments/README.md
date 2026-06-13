# TimeSorter 실험 결과 인덱스

한국어 일정 관리 특화 파인튜닝 파이프라인의 실험별 보고서 모음.

## 실험 목록

| 실험 | 환경 | 모델 | 데이터 | 최종 통과율 | 상태 |
|------|------|------|--------|------------|------|
| [RTX 12GB · Qwen3-4B v1-v5](RTX12G_Qwen3_4B_v1-v5/REPORT.md) | RTX 3080 Ti 12GB | Qwen3-4B-Instruct-2507 → Qwen3.5-4B | v1~v5 | **90.0%** (Qwen3.5) | ✅ 완료 |
| [RTX 4090 · Qwen3.5-9B v4](RTX4090_Qwen35_9B_v4/REPORT.md) | RTX 4090 24GB | Qwen3.5-9B | v4 (SFT+DPO) | 실행 대기 | 🔲 예정 |

## 핵심 결론

1. **성능 점프는 데이터 품질과 loss 마스킹에서** — DPO/GRPO 선호 학습 단독으로는 chain 약점 해결 불가
2. **chain 약점은 SFT 데이터 보강이 선행** — dependency_chain 43% → chain 특화 데이터 3× 증량 필요
3. **Qwen3.5 upgrade 효과** — 동일 데이터셋에서 4B Instruct 77.3% → Qwen3.5-4B 90.0% (+12.7%p)
