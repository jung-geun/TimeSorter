from __future__ import annotations

import os
import subprocess
import time
from typing import Literal

_FULL_GUIDE = (
    "우선순위 4축: 긴급도(마감 임박 여부) · 중요도(목표 기여도) · "
    "의존성(선행 필요 여부) · 시간 제약(소요 시간 vs 가용 시간). "
    "출력 포맷: '1) [할일] ([근거 태그]) 2) ...'"
)
_URGENCY_ONLY_GUIDE = (
    "마감이 가장 임박한 것을 최우선으로. 다른 기준은 무시. "
    "출력 포맷: '1) [할일] 2) ...'"
)
_NO_GUIDE = ""

# 모델 상수
_CLAUDE_SONNET = "claude-sonnet-4-6"
_CLAUDE_OPUS   = "claude-opus-4-6"
_CLAUDE_HAIKU  = "claude-haiku-4-5"

_OPENAI_HIGH   = "gpt-5.5"
_OPENAI_MINI   = "gpt-5.4-mini"

_GEMINI_FLASH  = "gemini-3.5-flash"
_GEMINI_PRO    = "gemini-3.1-pro"
_GEMINI_LITE   = "gemini-3.1-flash-lite"

_MODEL_ALIASES = {
    # Gemini
    "gemini-3.5-flash": "gemini-2.5-flash",       # 실제 프로덕션 모델로 맵핑
    "gemini-3.1-pro": "gemini-2.5-flash",         # 실제 프로덕션 모델로 맵핑
    "gemini-3.1-flash-lite": "gemini-1.5-flash",
    
    # Claude - CLI와 SDK가 직접 지원하므로 그대로 유지
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-haiku-4-5": "claude-haiku-4-5",
    
    # OpenAI
    "gpt-5.5": "gpt-4o",
    "gpt-5.4-mini": "gpt-4o-mini",
}

def _resolve_model(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name, model_name)

GuideType = Literal["full", "urgency_only", "none"]


def _guide(guide_type: GuideType) -> str:
    if guide_type == "full":
        return _FULL_GUIDE
    if guide_type == "urgency_only":
        return _URGENCY_ONLY_GUIDE
    return _NO_GUIDE


def _build_prompt(scenario: str, persona: str, guide_type: GuideType) -> str:
    guide = _guide(guide_type)
    if guide:
        return f"{guide}\n\n페르소나: {persona}\n할 일: {scenario}"
    return f"페르소나: {persona}\n할 일: {scenario}"


def _run_cli(cmd: list[str], timeout: int = 120) -> str:
    # /tmp에서 실행해 프로젝트 CLAUDE.md 컨텍스트를 차단
    env = os.environ.copy()
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd="/tmp", env=env, stdin=subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{cmd[0]} CLI failed")
    return result.stdout.strip()


def generate_with_claude(
    scenario: str, persona: str, guide_type: GuideType, model: str = _CLAUDE_SONNET
) -> str:
    """Claude로 응답 생성. API key가 있으면 SDK, 없으면 CLI fallback."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if os.environ.get("OPENAI_API_KEY"):
            fallback_model = _OPENAI_HIGH if "sonnet" in model or "opus" in model else _OPENAI_MINI
            print(f"  [폴백] ANTHROPIC_API_KEY가 없어 {model} 대신 OpenAI {fallback_model}을 사용합니다.")
            return generate_with_openai(scenario, persona, guide_type, model=fallback_model)

    model = _resolve_model(model)
    prompt = _build_prompt(scenario, persona, guide_type)
    for attempt in range(3):
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                msg = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            else:
                return _run_cli(["claude", "-p", prompt, "--model", model])
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def generate_with_openai(
    scenario: str, persona: str, guide_type: GuideType, model: str = _OPENAI_MINI
) -> str:
    """OpenAI로 응답 생성."""
    model = _resolve_model(model)
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = _build_prompt(scenario, persona, guide_type)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def generate_with_gemini(
    scenario: str, persona: str, guide_type: GuideType, model: str = _GEMINI_FLASH
) -> str:
    """Gemini로 응답 생성. API key가 있으면 SDK, 없으면 CLI fallback."""
    if not os.environ.get("GOOGLE_API_KEY"):
        prompt = _build_prompt(scenario, persona, guide_type)
        try:
            print(f"  [시도] GOOGLE_API_KEY가 없어 agy CLI로 {model} 생성을 시도합니다...")
            return _run_cli(["agy", "-p", prompt, "--dangerously-skip-permissions"], timeout=15)
        except Exception as e:
            print(f"  [실패] agy 호출 실패 또는 타임아웃 ({e})")

        if os.environ.get("OPENAI_API_KEY"):
            fallback_model = _OPENAI_HIGH if "3.5-flash" in model or "pro" in model else _OPENAI_MINI
            print(f"  [폴백] agy 사용 불가로 {model} 대신 OpenAI {fallback_model}을 사용합니다.")
            return generate_with_openai(scenario, persona, guide_type, model=fallback_model)

    model = _resolve_model(model)
    prompt = _build_prompt(scenario, persona, guide_type)
    for attempt in range(3):
        try:
            if os.environ.get("GOOGLE_API_KEY"):
                import google.generativeai as genai
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                gmodel = genai.GenerativeModel(model)
                return gmodel.generate_content(prompt).text.strip()
            else:
                return _run_cli(["gemini", "-p", prompt, "--model", model])
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def judge_pair(scenario: str, persona: str, a: str, b: str) -> Literal["A", "B", "TIE"]:
    """GPT-5로 두 응답을 비교해 A/B/TIE 반환."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"4축(긴급도·중요도·의존성·시간 제약) 기준으로 더 나은 우선순위 정렬 응답을 선택하세요.\n"
        f"페르소나: {persona}\n할 일: {scenario}\n\n"
        f"A: {a}\nB: {b}\n\n"
        "답: 'A', 'B', 'TIE' 중 하나만 출력."
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_resolve_model(_OPENAI_HIGH),
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            if verdict in ("A", "B", "TIE"):
                return verdict  # type: ignore[return-value]
            return "TIE"
        except Exception:
            if attempt == 2:
                return "TIE"
            time.sleep(2 ** attempt)
    return "TIE"


def generate_four_candidates(
    scenario: str, persona: str
) -> tuple[str, str, str, str]:
    """C1~C4 후보 4개 생성.

    C1: gemini-3.5-flash    full guide   (고품질 1)
    C2: claude-sonnet-4-6   full guide   (고품질 2)
    C3: gemini-3.1-lite     urgency only (편향)
    C4: claude-haiku-4-5    no guide     (저품질)
    """
    c1 = generate_with_gemini(scenario, persona, "full",         model=_GEMINI_FLASH)
    c2 = generate_with_claude(scenario, persona, "full",         model=_CLAUDE_SONNET)
    c3 = generate_with_gemini(scenario, persona, "urgency_only", model=_GEMINI_LITE)
    # API key 없으면 CLI가 코딩 에이전트로 거부하므로 OpenAI/Gemini fallback
    if os.environ.get("ANTHROPIC_API_KEY"):
        c4 = generate_with_claude(scenario, persona, "none", model=_CLAUDE_HAIKU)
    else:
        c4 = generate_with_openai(scenario, persona, "none", model=_OPENAI_MINI)
    return c1, c2, c3, c4
