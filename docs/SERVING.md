# TimeSorter — vLLM 서빙 및 이메일 파이프라인

## 개요

학습된 DPO 어댑터(`outputs/dpo_rtx12g_4b`)를 **vLLM OpenAI-compatible API**로 서빙합니다.
vLLM은 LoRA를 런타임에 동적으로 적용하므로 어댑터를 베이스 모델에 병합할 필요가 없습니다.

---

## 1. 아키텍처

```
[이메일 파일] ──► email_to_schedule.py
                    │
                    ├─ 태스크 추출: OpenAI gpt-4o-mini (JSON)
                    │
                    └─ 스케줄 생성: vLLM (Qwen3-4B + DPO 어댑터)
                                         │
                              OpenAI-compatible API
                              http://localhost:8000/v1/chat/completions
                                    model="scheduler"
```

---

## 2. 서버 기동

### Docker 방식 (권장)

```bash
# 서버 시작 (백그라운드 데몬)
make serve-docker

# 서버 중지
make serve-stop
```

내부적으로 실행되는 명령:

```bash
docker run -d --name timesorter-serve --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(PWD)/outputs:/workspace/outputs \
  -p 8000:8000 \
  -e LORA_PATH=outputs/dpo_rtx12g_4b \
  -e LORA_NAME=scheduler \
  -e GPU_MEM_UTIL=0.85 \
  vllm/vllm-openai:v0.8.5
```

`Dockerfile.serve`의 CMD:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --enable-lora \
    --lora-modules scheduler=outputs/dpo_rtx12g_4b \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --max-lora-rank 16 \
    --host 0.0.0.0 --port 8000
```

### 로컬 직접 실행 (vllm 설치 시)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --enable-lora \
    --lora-modules scheduler=outputs/dpo_rtx12g_4b \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --max-lora-rank 16 \
    --port 8000
```

---

## 3. 서버 상태 확인

서버 로드까지 **30~60초** 소요됩니다 (torch.compile 포함 첫 기동 시 ~90초).

```bash
# 헬스체크
curl http://localhost:8000/health
# → {"status":"healthy"}

# 모델 목록
curl http://localhost:8000/v1/models | python3 -m json.tool
# → {"data": [{"id": "scheduler", ...}]}
```

또는 `scripts/serve.py` 유틸리티 사용:

```bash
# 헬스체크
uv run python scripts/serve.py --health-check

# 서버 준비될 때까지 대기 (최대 120초)
uv run python scripts/serve.py --wait

# 모델 목록 출력
uv run python scripts/serve.py --list-models
```

---

## 4. API 직접 호출

OpenAI Python SDK 또는 `curl`로 호출 가능합니다. `api_key="EMPTY"`를 사용합니다.

### curl 예시

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "scheduler",
    "messages": [
      {
        "role": "user",
        "content": "[직장인 씨의 오늘의 할 일 목록]\n- 보고서 작성 (내일 오전 마감)\n- 팀 미팅 (오후 2시)\n- 메일 답장\n- 코드 리뷰"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.1
  }' | python3 -m json.tool
```

### Python SDK 예시

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="scheduler",
    messages=[
        {
            "role": "user",
            "content": "[직장인 씨의 오늘의 할 일 목록]\n- 보고서 작성 (내일 오전 마감)\n- 팀 미팅 (오후 2시)\n- 메일 답장",
        }
    ],
    max_tokens=512,
    temperature=0.1,
)
print(response.choices[0].message.content)
```

### 기대 출력 형식

```
1) 팀 미팅 - 오후 2시 고정 일정, 시간 제약↑
2) 보고서 작성 - 내일 오전 마감, 긴급도↑ 준비 시간 필요
3) 코드 리뷰 - 팀 블로킹 작업, 의존성↑
4) 메일 답장 - 마감 없음, 다른 작업 후 처리 가능
```

---

## 5. 이메일 파이프라인

이메일 파일 5건(`data/sample_emails/`)을 순차 처리하여 통합 스케줄을 생성합니다.

### 전체 파이프라인 실행

```bash
# 서버가 실행 중이어야 합니다
make email-pipeline

# 또는 직접
uv run python scripts/email_to_schedule.py \
    --email-dir data/sample_emails \
    --persona 직장인 \
    --server-url http://localhost:8000 \
    --model scheduler \
    --out outputs/schedule_result.json
```

### 태스크 추출만 (서버 불필요)

```bash
make email-extract

# 또는
uv run python scripts/email_to_schedule.py \
    --email-dir data/sample_emails \
    --persona 직장인 \
    --extract-only
```

### 파이프라인 + 검증 통합

```bash
make validate-and-pipeline
```

### 출력 형식 (`outputs/schedule_result.json`)

```json
{
  "persona": "직장인",
  "tasks": [
    "그린테크 파트너십 미팅 참석 가능 여부 확인 및 회신",
    "오늘 오후 2시 파트너십 미팅 참석 (3층 대회의실)",
    ...
  ],
  "schedule": "1) 항목 - 이유\n2) ...",
  "email_count": 5
}
```

---

## 6. vLLM 서버 성능 특성

| 항목 | 값 |
|------|-----|
| 기반 이미지 | `vllm/vllm-openai:v0.8.5` |
| 최대 KV 캐시 | 8,016 tokens (`--max-model-len 2048`) |
| 최대 동시 요청 | ~3.91× (vLLM 추정) |
| 첫 기동 시간 | ~52s (torch.compile) + ~40s (그래프 캡처) |
| 이후 기동 시간 | ~30s (컴파일 캐시 히트) |
| VRAM 사용량 | ~10.2GB (GPU_MEM_UTIL=0.85, 12GB 카드) |
| dtype | bfloat16 |

---

## 7. 주요 플래그 설명

| 플래그 | 값 | 이유 |
|--------|-----|------|
| `--enable-lora` | — | LoRA 어댑터 런타임 적용 활성화 |
| `--max-lora-rank 16` | 16 | 학습된 어댑터 r=16과 일치 |
| `--gpu-memory-utilization` | 0.85 | KV 캐시와 모델 가중치 간 균형 |
| `--max-model-len` | 2048 | 학습 시 max_seq_length와 동일 |
| `--dtype bfloat16` | — | 학습 compute dtype과 일치, fp16 대비 안정 |

---

## 8. 교차 검증 실행

```bash
# 기본 (gpt-5.5 judge)
make validate

# 옵션 지정
uv run python scripts/validate_schedule.py \
    --result outputs/schedule_result.json \
    --email-dir data/sample_emails \
    --judge gpt-5.5 \
    --out outputs/validation_result.json
```
