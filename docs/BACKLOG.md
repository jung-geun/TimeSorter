# TimeSorter — 개발 백로그

## 완료된 작업

### v0.1 — 기반 구축
- [x] Qwen3-4B-Instruct-2507 기반 SFT/DPO 학습 파이프라인 설계
- [x] `src/drl/` 모듈 구조 (config, model, data, device, train_sft, train_dpo, infer)
- [x] NF4 QLoRA 4-bit 양자화 적용 (12GB VRAM RTX 환경)
- [x] flash-attn 2.7.4 Docker 환경 구축 (`timesorter:cu124`)

### v0.2 — 데이터셋 확장
- [x] `events-scheduling` 시드 → Nemotron-Personas-Korea 페르소나 4-round 생성 (2,000개)
- [x] gen_korean_schedule.py 페르소나 4→8개 확장 (4,000개)
- [x] ko_Ultrafeedback_binarized 스케줄링 키워드 필터 혼합 (런타임 500개)
- [x] merge_datasets.py 병합 + 중복 제거 → scheduler_ko_combined.parquet (5,999개)
- [x] Git LFS 마이그레이션 (data/*.parquet, *.jsonl)

### v0.3 — SFT 학습
- [x] sft_rtx12g_4b.yaml 구성 (batch 1, grad_accum 32, adamw_8bit, packing)
- [x] Docker 학습 (`make sft-docker`) — 225 steps, 5 epoch, ~3시간
- [x] SFT 결과: train_loss=1.179, mean_token_accuracy=77.55%
- [x] 어댑터 저장: `outputs/sft_rtx12g_4b/`

### v0.4 — DPO 데이터셋 + 학습
- [x] gen_preference_pairs.py 비동기화 (asyncio, 7x 속도 향상)
- [x] 4-후보 생성 (Gemini-full / Claude-full / Gemini-urgency / Claude-noguide) + GPT judge
- [x] OpenAI fallback (Anthropic 401, Google 미설정 시 자동 전환)
- [x] dpo_pairs.parquet 생성: 500시나리오 × 3콤보 = 1,469쌍
- [x] dpo_rtx12g_4b.yaml 구성 (ref_model=None PEFT trick, beta=0.1)
- [x] DPO 학습 (`make dpo-docker`) — 92 steps, 2 epoch, ~39분
- [x] DPO 결과: train_loss=0.117, reward_accuracy peak 98.8%
- [x] 어댑터 저장: `outputs/dpo_rtx12g_4b/`

### v0.5 — 서빙 + 이메일 파이프라인
- [x] Dockerfile.serve (vllm/vllm-openai:v0.8.5 기반)
- [x] vLLM LoRA 서빙 (`make serve-docker`, port 8000)
- [x] email_to_schedule.py: 이메일 → 태스크 추출 → vLLM 스케줄 파이프라인
- [x] 샘플 이메일 5건 (보고서/미팅/계약/점심/코드리뷰)
- [x] validate_schedule.py: 2-phase 교차 검증 (gpt-5.5 judge)
- [x] `make validate-and-pipeline` 통합 타겟

## 진행 예정

### v0.6 — 모델 품질 개선
- [ ] DPO 데이터셋 확장 (현재 500 → 2,000 시나리오)
- [ ] 오후 2시 미팅 환각 버그 수정: 시간 제약 명시 프롬프트 추가
- [ ] 마감일 파싱 정확도 향상 (오늘/내일/X월 Y일 → 절대 날짜 변환)
- [ ] 멀티턴 대화 지원 (수정 요청 시 재정렬)

### v0.7 — 인프라
- [ ] HuggingFace Hub 어댑터 업로드
- [ ] vLLM torch.compile 캐시 지속 (재기동 시간 20s → 2s)
- [ ] 복수 판사 모델 지원 (Claude Sonnet, Gemini Flash 병렬 검증)
- [ ] 정량 벤치마크 (hold-out 100건, NDCG@k 우선순위 지표)

### v0.8 — 프로덕션
- [ ] FastAPI 게이트웨이 (이메일 수신 → 스케줄 자동 push)
- [ ] 사용자별 페르소나 프로파일 저장
- [ ] 달력 연동 (Google Calendar API)
