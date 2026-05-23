#!/usr/bin/env python
"""이메일 디렉토리를 읽어 할 일을 추출한 뒤 학습된 스케줄 모델로 우선순위를 정렬합니다.

동작 흐름:
  1. 이메일 파일(txt/eml) 시퀀셜 파싱
  2. OpenAI gpt-5.4-mini로 각 이메일에서 태스크 추출 (할 일 + 마감)
  3. 추출된 태스크를 하나의 목록으로 통합
  4. vLLM (로컬 OpenAI 호환 API)에 스케줄 우선순위 요청
  5. 결과 출력 및 JSON 저장

사용:
  # vLLM 서버가 실행 중이어야 합니다 (make serve-docker)
  uv run python scripts/email_to_schedule.py

  # 디렉토리 지정
  uv run python scripts/email_to_schedule.py --email-dir data/sample_emails

  # 서버 URL 지정 (기본: http://localhost:8000)
  uv run python scripts/email_to_schedule.py --server-url http://localhost:8000

  # 태스크 추출만 (스케줄링 없이)
  uv run python scripts/email_to_schedule.py --extract-only

  # 출력 저장
  uv run python scripts/email_to_schedule.py --out outputs/schedule_result.json
"""
from __future__ import annotations

import argparse
import email
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from drl.data.schema import (
    SCHEDULER_SYSTEM_PROMPT_V1,
    SCHEDULER_SYSTEM_PROMPT_V2,
    parse_or_repair,
    render_system_prompt,
    response_to_text,
)

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """당신은 이메일에서 '수신자가 처리해야 할 행동 항목(task)'을 추출하는 전문가입니다.
이메일 본문을 읽고, 수신자가 오늘 또는 가까운 시일 내에 처리해야 할 일을 정리하세요.

규칙:
- 각 태스크는 한 줄로 간결하게 (동사형 종결: "~하기", "~제출", "~참석" 등)
- 마감이 명시된 경우 괄호 안에 포함 (예: "계약서 검토 (5/26 오전까지)")
- 명확한 행동이 없는 단순 정보성 이메일은 tasks를 빈 배열로 반환
- 반드시 {"tasks": ["...", "..."]} 형식의 JSON으로만 응답"""

_EXTRACT_USER = """다음 이메일에서 수신자가 처리해야 할 행동 항목을 추출하세요.
반드시 {{"tasks": ["태스크1", "태스크2", ...]}} 형식으로 응답하세요.

{email_content}"""

_SCHEDULE_PROMPT_TEMPLATE = "[{persona}의 오늘의 할 일 목록]\n{task_lines}"

DEFAULT_PERSONA = "직장인"
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "scheduler"  # vLLM --served-model-name
EXTRACTOR_MODEL = "gpt-5.4-mini"
DEFAULT_SCHEMA_VERSION = "v1"


# ── 데이터 구조 ───────────────────────────────────────────────────────────────

@dataclass
class EmailTask:
    source_file: str
    subject: str
    tasks: list[str]


@dataclass
class ScheduleResult:
    persona: str
    tasks: list[str]
    schedule: str
    sources: list[str]


# ── 이메일 파싱 ───────────────────────────────────────────────────────────────

def _parse_email_file(path: Path) -> tuple[str, str]:
    """(subject, body) 반환."""
    raw = path.read_text(encoding="utf-8", errors="replace")

    # .eml 형식 파싱
    if path.suffix == ".eml" or raw.startswith(("From:", "MIME-Version:")):
        msg = email.message_from_string(raw)
        subject = msg.get("Subject", path.stem)
        if msg.is_multipart():
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            body = payload.decode("utf-8", errors="replace") if payload else msg.get_payload()
        return subject, body

    # .txt 헤더 파싱 (From:/To:/Subject:/Date: 형식)
    lines = raw.splitlines()
    subject = path.stem
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("Subject:"):
            subject = line.split(":", 1)[1].strip()
        elif line.strip() == "" and i > 0:
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:])
    return subject, body


# ── 태스크 추출 ───────────────────────────────────────────────────────────────

def extract_tasks_from_email(
    subject: str,
    body: str,
    client: OpenAI,
    source_file: str = "",
) -> EmailTask:
    content = f"제목: {subject}\n\n{body.strip()}"
    try:
        resp = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": _EXTRACT_USER.format(email_content=content)},
            ],
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )
        raw_json = resp.choices[0].message.content.strip()
        # JSON object or array 처리
        parsed = json.loads(raw_json)
        if isinstance(parsed, list):
            tasks = [str(t) for t in parsed if t]
        elif isinstance(parsed, dict):
            # {"tasks": [...]} 또는 다른 키명 대응 — 첫 번째 list 값 사용
            for v in parsed.values():
                if isinstance(v, list):
                    tasks = [str(t) for t in v if t]
                    break
            else:
                tasks = []
        else:
            tasks = []
    except Exception as e:
        print(f"  [경고] 태스크 추출 실패 ({source_file}): {e}", file=sys.stderr)
        tasks = []

    return EmailTask(source_file=source_file, subject=subject, tasks=tasks)


# ── 스케줄 생성 ───────────────────────────────────────────────────────────────

def generate_schedule(
    tasks: list[str],
    persona: str,
    server_url: str,
    model_name: str,
    schema_version: str = "v1",
) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    user_content = _SCHEDULE_PROMPT_TEMPLATE.format(
        persona=persona,
        task_lines=task_lines,
    )

    system_tmpl = (
        SCHEDULER_SYSTEM_PROMPT_V2 if schema_version == "v2" else SCHEDULER_SYSTEM_PROMPT_V1
    )
    messages = [
        {"role": "system", "content": render_system_prompt(system_tmpl, persona)},
        {"role": "user", "content": user_content},
    ]

    create_kwargs: dict = dict(
        model=model_name,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    if schema_version == "v2":
        create_kwargs["extra_body"] = {"guided_json": True}

    client = OpenAI(
        api_key="EMPTY",  # vLLM은 키 검증 없음
        base_url=f"{server_url}/v1",
    )
    resp = client.chat.completions.create(**create_kwargs)
    raw = resp.choices[0].message.content.strip()

    if schema_version == "v2":
        parsed = parse_or_repair(raw)
        return response_to_text(parsed)
    return raw


# ── 메인 파이프라인 ───────────────────────────────────────────────────────────

def run_pipeline(
    email_dir: Path,
    persona: str,
    server_url: str,
    model_name: str,
    extract_only: bool,
    openai_client: OpenAI,
    schema_version: str = "v1",
    verbose: bool = True,
) -> ScheduleResult:
    email_files = sorted(
        p for p in email_dir.iterdir()
        if p.suffix in (".txt", ".eml") and not p.name.startswith(".")
    )
    if not email_files:
        print(f"[에러] {email_dir}에 .txt/.eml 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  이메일 → 스케줄 파이프라인")
    print(f"  이메일 {len(email_files)}건 | 페르소나: {persona}")
    print(f"{'='*60}\n")

    # Step 1: 이메일별 태스크 추출
    all_email_tasks: list[EmailTask] = []
    for path in email_files:
        subject, body = _parse_email_file(path)
        if verbose:
            print(f"[{path.name}] 제목: {subject[:50]}")
        et = extract_tasks_from_email(subject, body, openai_client, source_file=path.name)
        if et.tasks:
            all_email_tasks.append(et)
            for t in et.tasks:
                print(f"  → {t}")
        else:
            print("  → (추출된 태스크 없음)")
        print()

    # Step 2: 전체 태스크 통합 (이메일 순서 유지)
    all_tasks: list[str] = []
    sources: list[str] = []
    for et in all_email_tasks:
        all_tasks.extend(et.tasks)
        sources.append(et.source_file)

    if not all_tasks:
        print("[경고] 추출된 태스크가 없습니다.")
        return ScheduleResult(persona=persona, tasks=[], schedule="", sources=[])

    print(f"{'─'*60}")
    print(f"  총 {len(all_tasks)}개 태스크 추출됨")
    print(f"{'─'*60}\n")

    if extract_only:
        return ScheduleResult(persona=persona, tasks=all_tasks, schedule="", sources=sources)

    # Step 3: vLLM으로 스케줄 우선순위 생성
    print(f"[스케줄링] {server_url} ({model_name}) 호출 중...")
    schedule = generate_schedule(all_tasks, persona, server_url, model_name, schema_version)

    print(f"\n{'='*60}")
    print("  우선순위 스케줄 결과")
    print(f"{'='*60}")
    print(schedule)
    print(f"{'='*60}\n")

    return ScheduleResult(persona=persona, tasks=all_tasks, schedule=schedule, sources=sources)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이메일 → 스케줄 파이프라인")
    parser.add_argument(
        "--email-dir", default="data/sample_emails",
        help="이메일 파일 디렉토리 (기본: data/sample_emails)"
    )
    parser.add_argument(
        "--persona", default=DEFAULT_PERSONA,
        help=f"사용자 페르소나 (기본: {DEFAULT_PERSONA})"
    )
    parser.add_argument(
        "--server-url", default=DEFAULT_SERVER_URL,
        help=f"vLLM 서버 URL (기본: {DEFAULT_SERVER_URL})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME,
        help=f"vLLM 서빙 모델명 (기본: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="태스크 추출만 수행 (vLLM 서버 불필요)"
    )
    parser.add_argument(
        "--out", default=None,
        help="결과 JSON 저장 경로 (기본: 미저장)"
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="vLLM 서버 대기 없이 바로 시도"
    )
    parser.add_argument(
        "--schema-version", default=DEFAULT_SCHEMA_VERSION, choices=["v1", "v2"],
        help="출력 스키마 버전 (v1=자유 텍스트, v2=JSON+4축 점수)"
    )
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("[에러] OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)

    # vLLM 서버 대기 (extract-only 모드에서는 불필요)
    if not args.extract_only and not args.no_wait:
        from serve import wait_for_server
        if not wait_for_server(args.server_url, max_wait=60):
            print(
                f"[에러] vLLM 서버({args.server_url})에 연결할 수 없습니다.\n"
                "  'make serve-docker'로 서버를 먼저 기동하거나 --extract-only 옵션을 사용하세요.",
                file=sys.stderr,
            )
            sys.exit(1)

    result = run_pipeline(
        email_dir=Path(args.email_dir),
        persona=args.persona,
        server_url=args.server_url,
        model_name=args.model,
        extract_only=args.extract_only,
        openai_client=openai_client,
        schema_version=args.schema_version,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[저장] {out_path}")
