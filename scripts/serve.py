#!/usr/bin/env python
"""vLLM 서버 런처 및 유틸리티.

사용:
  # Docker로 서버 기동 (권장)
  make serve-docker

  # 직접 실행 (vllm 설치 필요)
  uv run python scripts/serve.py --adapter outputs/dpo_rtx12g_4b

  # 헬스체크
  uv run python scripts/serve.py --health-check

  # 모델 목록 확인
  uv run python scripts/serve.py --list-models
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import httpx


BASE_URL = "http://localhost:8000"


def health_check(url: str = BASE_URL, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_server(url: str = BASE_URL, max_wait: int = 120) -> bool:
    print(f"[서버 대기] {url} 응답 기다리는 중 (최대 {max_wait}초)...")
    for i in range(max_wait):
        if health_check(url):
            print(f"[OK] 서버 준비 완료 ({i+1}초 소요)")
            return True
        time.sleep(1)
        if i % 10 == 9:
            print(f"  ...{i+1}초 경과")
    print("[실패] 서버 응답 없음")
    return False


def list_models(url: str = BASE_URL) -> None:
    try:
        r = httpx.get(f"{url}/v1/models", timeout=10)
        r.raise_for_status()
        for m in r.json()["data"]:
            print(f"  - {m['id']}")
    except Exception as e:
        print(f"[에러] 모델 목록 조회 실패: {e}")


def serve_local(
    adapter: str,
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    lora_name: str = "scheduler",
    max_model_len: int = 2048,
    gpu_mem_util: float = 0.85,
    port: int = 8000,
) -> None:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--enable-lora",
        "--lora-modules", f"{lora_name}={adapter}",
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-lora-rank", "16",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", lora_name,
    ]
    print(f"[서버 기동] {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/dpo_rtx12g_4b")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--lora-name", default="scheduler")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--wait", action="store_true", help="서버 기동 대기")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"

    if args.health_check:
        ok = health_check(url)
        print("OK" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    elif args.wait:
        ok = wait_for_server(url)
        sys.exit(0 if ok else 1)
    elif args.list_models:
        list_models(url)
    else:
        serve_local(
            adapter=args.adapter,
            model=args.model,
            lora_name=args.lora_name,
            port=args.port,
            gpu_mem_util=args.gpu_mem_util,
        )
