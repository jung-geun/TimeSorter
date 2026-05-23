#!/usr/bin/env python
"""거부 케이스 합성 — Phase B.

도메인 무관 입력(레시피, 수학 문제, 잡담 등)에 대해 모델이
{"tasks": [], "refusal_reason": "..."} 로 응답하도록 학습하는 예제를 생성한다.

API 호출 없이 템플릿 확장으로 생성 ($0).

출력:
  data/refusals_sft_v2.parquet   — SFT용 (prompt / chosen / persona / source)
  data/refusals_dpo_v2.parquet   — DPO용 (prompt / chosen / rejected / persona / source)

사용:
  uv run python scripts/synthesize_refusals.py
  uv run python scripts/synthesize_refusals.py --limit 20 --verify
  uv run python scripts/synthesize_refusals.py --sft-out data/refusals_sft_v2.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from timesorter.data.schema import (
    ScheduleResponse,
    parse_lenient,
)

# ── 8 페르소나 ────────────────────────────────────────────────────────────────

_PERSONAS = [
    "직장인",
    "학생",
    "프리랜서",
    "부모",
    "시니어",
    "창업자",
    "의료진",
    "연구자",
]

# ── 30종 비-태스크 입력 패턴 ──────────────────────────────────────────────────
# (카테고리, 입력 템플릿 목록)

_PATTERNS: list[tuple[str, list[str]]] = [
    ("recipe", [
        "불고기 레시피 알려줘",
        "김치찌개 만드는 법이 뭐야?",
        "파스타 요리 순서를 알려줘",
        "된장국 끓이는 방법 설명해줘",
        "계란말이 레시피 좀 알려줘",
    ]),
    ("math", [
        "1부터 100까지 더하면 얼마야?",
        "이차방정식 x²-5x+6=0 풀어줘",
        "원의 넓이 공식은 뭐야?",
        "피타고라스 정리로 빗변 구하는 법 알려줘",
        "소수의 정의가 뭐야?",
    ]),
    ("weather", [
        "오늘 서울 날씨 어때?",
        "이번 주 비 오나요?",
        "내일 기온이 몇 도야?",
        "미세먼지 농도 알 수 있어?",
        "태풍이 오고 있나요?",
    ]),
    ("creative_writing", [
        "봄에 관한 시 한 편 써줘",
        "단편 소설 도입부 써줘",
        "사랑에 관한 짧은 에세이 써줘",
        "고양이를 주제로 동화 써줘",
        "자기소개서 첫 문단 써줘",
    ]),
    ("coding", [
        "파이썬으로 피보나치 수열 짜줘",
        "리스트 정렬하는 코드 알려줘",
        "SQL WHERE 절 사용법 설명해줘",
        "HTML 테이블 만드는 법 알려줘",
        "자바스크립트로 타이머 만들기",
    ]),
    ("history", [
        "조선왕조 건국 연도가 언제야?",
        "임진왜란 원인이 뭐야?",
        "세계대전은 몇 번 있었어?",
        "이순신 장군의 업적은?",
        "고려시대 문화재 알려줘",
    ]),
    ("greeting", [
        "안녕하세요!",
        "오늘 하루도 좋은 날 되세요",
        "반갑습니다, 처음 뵙겠습니다",
        "좋은 아침이에요!",
        "잘 부탁드립니다~",
    ]),
    ("translation", [
        "'사랑해'를 영어로 번역해줘",
        "How are you?를 한국어로",
        "'감사합니다'를 일본어로",
        "Merci를 한국어로 번역",
        "'안녕'을 중국어로 번역",
    ]),
    ("news", [
        "최근 주요 뉴스 요약해줘",
        "어제 경제 뉴스 알려줘",
        "최신 IT 뉴스 뭐가 있어?",
        "오늘 스포츠 경기 결과는?",
        "최근 환율 동향 알려줘",
    ]),
    ("sports", [
        "야구 규칙 설명해줘",
        "축구 오프사이드가 뭐야?",
        "농구 자유투 규칙은?",
        "테니스 점수 매기는 법",
        "수영 영법 종류 알려줘",
    ]),
    ("music", [
        "재즈와 블루스 차이가 뭐야?",
        "피아노 초보자용 곡 추천해줘",
        "클래식 명곡 Top 5는?",
        "K-POP 최신 곡 추천",
        "기타 코드 F 잡는 법",
    ]),
    ("movie", [
        "오늘 볼 만한 영화 추천해줘",
        "공포 영화 명작이 뭐가 있어?",
        "최근 개봉 한국 영화는?",
        "넷플릭스 추천 드라마 알려줘",
        "스릴러 영화 베스트 5",
    ]),
    ("travel", [
        "제주도 여행 코스 알려줘",
        "유럽 배낭여행 팁이 있어?",
        "일본 여행 필수 코스는?",
        "비행기 저렴하게 예약하는 법",
        "서울 당일치기 여행 코스",
    ]),
    ("medical", [
        "감기와 독감 차이가 뭐야?",
        "두통이 심할 때 어떻게 해?",
        "혈압 정상 범위가 어떻게 돼?",
        "비타민 D 부족 증상은?",
        "소화불량에 좋은 음식은?",
    ]),
    ("legal", [
        "계약서 작성 시 주의사항은?",
        "저작권법 위반이 뭐야?",
        "부동산 등기 절차 알려줘",
        "이혼 소송 절차가 어떻게 돼?",
        "교통사고 처리 방법은?",
    ]),
    ("philosophy", [
        "자유의지란 무엇인가요?",
        "행복의 정의는 뭐야?",
        "소크라테스의 철학 핵심은?",
        "공리주의와 의무론 차이는?",
        "실존주의란 무엇인가?",
    ]),
    ("math_word", [
        "사과 3개에 2000원이면 10개는?",
        "기차가 시속 80km로 달릴 때 2시간 후 거리는?",
        "할인율 30%면 10만원짜리 얼마야?",
        "정사각형 한 변이 5cm일 때 넓이는?",
        "평균 점수: 80, 90, 70, 85이면?",
    ]),
    ("geography", [
        "한국의 수도는 어디야?",
        "나일강 길이가 얼마야?",
        "에베레스트 산 높이는?",
        "아마존 강은 어느 나라에 있어?",
        "북극과 남극 차이는 뭐야?",
    ]),
    ("science", [
        "광합성 과정을 설명해줘",
        "DNA와 RNA 차이가 뭐야?",
        "블랙홀이 뭐야?",
        "상대성 이론을 쉽게 설명해줘",
        "산화와 환원 반응이 뭐야?",
    ]),
    ("pet", [
        "강아지 훈련 기초 방법은?",
        "고양이 밥 하루에 얼마나 줘야 해?",
        "토끼 키울 때 주의사항은?",
        "강아지가 짖지 않게 하는 법",
        "고양이 중성화 수술 시기는?",
    ]),
    ("finance", [
        "주식 초보자가 알아야 할 것은?",
        "ETF와 주식 차이가 뭐야?",
        "연금저축 가입 방법은?",
        "재테크 기초 알려줘",
        "부동산 투자 시 주의점은?",
    ]),
    ("career", [
        "이직할 때 협상 팁 알려줘",
        "포트폴리오 작성법은?",
        "면접 단골 질문과 답변은?",
        "연봉 협상하는 법 알려줘",
        "직무 분석 어떻게 해?",
    ]),
    ("relationship", [
        "친구와 다퉜을 때 화해하는 법",
        "소개팅 첫 만남 대화 주제는?",
        "직장 내 갈등 해결 방법은?",
        "부모님께 진로 말하는 법",
        "부부 싸움 후 화해하는 방법",
    ]),
    ("diy", [
        "벽에 못 박는 법 알려줘",
        "의자 수리하는 방법은?",
        "IKEA 가구 조립 팁이 있어?",
        "방 페인트 칠하는 방법",
        "창문 방음하는 DIY 방법",
    ]),
    ("game", [
        "체스 기본 규칙 알려줘",
        "배드민턴 서브 규칙은?",
        "포커 족보 알려줘",
        "바둑 입문하는 방법은?",
        "스크래블 게임 방법은?",
    ]),
    ("fashion", [
        "정장 코디 기본 원칙은?",
        "캐주얼 룩 기본 아이템은?",
        "색 조합 잘하는 법 알려줘",
        "계절별 필수 아이템은?",
        "빈티지 패션이란 뭐야?",
    ]),
    ("trivia", [
        "세계에서 가장 큰 동물은?",
        "인간의 뼈 개수가 몇 개야?",
        "커피는 어느 나라에서 유래했어?",
        "한글 창제 연도가 언제야?",
        "달에서 지구까지 거리는?",
    ]),
    ("joke", [
        "재미있는 아재개그 하나 해줘",
        "짧고 웃긴 농담 알려줘",
        "수학 관련 유머 있어?",
        "직장인 공감 개그 해줘",
        "고양이 관련 유머 하나 해줘",
    ]),
    ("small_talk", [
        "오늘 기분이 좀 안 좋아",
        "심심한데 뭐 하면 좋을까?",
        "요즘 너무 피곤해서 힘들어",
        "넌 AI라서 뭐가 좋아?",
        "취미가 뭐야?",
    ]),
    ("random_question", [
        "우주는 끝이 있어?",
        "시간 여행이 가능할까?",
        "외계인이 존재할까?",
        "인간이 달에 다시 갈 수 있을까?",
        "AI가 인간보다 똑똑해질 수 있을까?",
    ]),
]

assert len(_PATTERNS) == 30, f"패턴이 30개 아님: {len(_PATTERNS)}"

# ── 거부 응답 생성 ────────────────────────────────────────────────────────────

_REFUSAL_REASONS: dict[str, list[str]] = {
    "recipe": [
        "입력이 요리 레시피 요청입니다. 할 일 목록이 아니므로 우선순위를 정렬할 수 없습니다.",
        "레시피 관련 질문은 우선순위 정렬 비서의 도메인이 아닙니다.",
    ],
    "math": [
        "수학 문제 풀이 요청입니다. 할 일 목록을 입력해 주세요.",
        "수학 계산은 지원하지 않습니다. 처리해야 할 업무 목록을 입력해 주세요.",
    ],
    "weather": [
        "날씨 정보 요청입니다. 오늘의 할 일 목록을 입력해 주세요.",
        "날씨 서비스는 제공하지 않습니다. 할 일 목록을 입력해 주시면 우선순위를 정렬해 드립니다.",
    ],
    "creative_writing": [
        "창작 요청입니다. 할 일 목록이 아니어서 처리할 수 없습니다.",
        "글쓰기 도움은 제 전문이 아닙니다. 처리해야 할 업무를 알려주세요.",
    ],
    "coding": [
        "코딩 질문입니다. 할 일 목록을 입력하면 우선순위를 정렬해 드립니다.",
        "프로그래밍 도움은 지원하지 않습니다. 업무 목록을 입력해 주세요.",
    ],
    "history": [
        "역사 관련 질문입니다. 할 일 목록이 아닙니다.",
        "역사 정보 제공은 지원 범위 밖입니다. 처리할 업무 목록을 입력해 주세요.",
    ],
    "greeting": [
        "인사말입니다. 처리해야 할 할 일 목록을 입력해 주세요.",
        "안녕하세요! 오늘의 할 일 목록을 알려주시면 우선순위를 정렬해 드리겠습니다.",
    ],
    "translation": [
        "번역 요청입니다. 할 일 목록이 아니므로 처리할 수 없습니다.",
        "번역 서비스는 제공하지 않습니다. 오늘의 업무 목록을 입력해 주세요.",
    ],
    "news": [
        "뉴스 요청입니다. 할 일 목록을 입력해 주세요.",
        "뉴스 정보는 제공하지 않습니다. 처리해야 할 업무가 있으면 알려주세요.",
    ],
    "sports": [
        "스포츠 규칙 질문입니다. 할 일 목록을 입력해 주시면 우선순위 정렬을 도와드립니다.",
        "스포츠 정보는 지원 범위 밖입니다. 오늘의 업무 목록을 알려주세요.",
    ],
    "music": [
        "음악 관련 질문입니다. 할 일 목록이 아닙니다.",
        "음악 추천은 지원하지 않습니다. 처리해야 할 업무를 입력해 주세요.",
    ],
    "movie": [
        "영화 추천 요청입니다. 할 일 목록을 입력해 주세요.",
        "영화 정보는 제공하지 않습니다. 오늘의 할 일 목록을 알려주세요.",
    ],
    "travel": [
        "여행 관련 질문입니다. 할 일 목록이 아닙니다.",
        "여행 정보는 지원 범위 밖입니다. 처리해야 할 업무 목록을 입력해 주세요.",
    ],
    "medical": [
        "의료 관련 질문입니다. 할 일 목록을 입력해 주세요.",
        "의료 정보는 제공하지 않습니다. 처리해야 할 업무가 있으면 알려주세요.",
    ],
    "legal": [
        "법률 관련 질문입니다. 할 일 목록이 아닙니다.",
        "법률 정보는 지원 범위 밖입니다. 오늘의 할 일 목록을 입력해 주세요.",
    ],
    "philosophy": [
        "철학적 질문입니다. 할 일 목록이 아닙니다.",
        "철학 토론은 지원하지 않습니다. 처리해야 할 업무를 알려주세요.",
    ],
    "math_word": [
        "수학 응용 문제입니다. 할 일 목록을 입력해 주세요.",
        "수학 풀이는 지원하지 않습니다. 처리해야 할 업무 목록을 입력해 주세요.",
    ],
    "geography": [
        "지리 관련 질문입니다. 할 일 목록이 아닙니다.",
        "지리 정보는 제공하지 않습니다. 오늘의 업무 목록을 알려주세요.",
    ],
    "science": [
        "과학 관련 질문입니다. 할 일 목록이 아닙니다.",
        "과학 정보는 지원 범위 밖입니다. 처리해야 할 업무를 입력해 주세요.",
    ],
    "pet": [
        "반려동물 관련 질문입니다. 할 일 목록을 입력해 주세요.",
        "반려동물 정보는 지원하지 않습니다. 오늘의 업무 목록을 알려주세요.",
    ],
    "finance": [
        "재테크 관련 질문입니다. 할 일 목록이 아닙니다.",
        "금융 정보는 지원 범위 밖입니다. 처리해야 할 업무를 알려주세요.",
    ],
    "career": [
        "경력 관련 질문입니다. 할 일 목록을 입력해 주세요.",
        "취업·이직 정보는 지원하지 않습니다. 처리해야 할 업무 목록을 입력해 주세요.",
    ],
    "relationship": [
        "관계 관련 상담 요청입니다. 할 일 목록이 아닙니다.",
        "상담 서비스는 지원하지 않습니다. 처리해야 할 업무를 알려주세요.",
    ],
    "diy": [
        "DIY 관련 질문입니다. 할 일 목록을 입력해 주세요.",
        "DIY 정보는 지원 범위 밖입니다. 오늘의 업무 목록을 알려주세요.",
    ],
    "game": [
        "게임 관련 질문입니다. 할 일 목록이 아닙니다.",
        "게임 규칙 안내는 지원하지 않습니다. 처리해야 할 업무를 입력해 주세요.",
    ],
    "fashion": [
        "패션 관련 질문입니다. 할 일 목록을 입력해 주세요.",
        "패션 조언은 지원하지 않습니다. 처리해야 할 업무 목록을 입력해 주세요.",
    ],
    "trivia": [
        "상식 퀴즈 질문입니다. 할 일 목록이 아닙니다.",
        "상식 정보는 제공하지 않습니다. 오늘의 업무 목록을 알려주세요.",
    ],
    "joke": [
        "유머 요청입니다. 할 일 목록이 아닙니다.",
        "농담 서비스는 지원하지 않습니다. 처리해야 할 업무를 입력해 주세요.",
    ],
    "small_talk": [
        "일상적인 대화 요청입니다. 할 일 목록을 입력해 주세요.",
        "잡담은 지원 범위 밖입니다. 처리해야 할 업무가 있으면 알려주세요.",
    ],
    "random_question": [
        "일반적인 질문입니다. 할 일 목록이 아닙니다.",
        "일반 질문 응답은 지원하지 않습니다. 처리해야 할 업무를 입력해 주세요.",
    ],
}

# ── rejected 응답 (DPO용: 거부해야 하는데 태스크로 처리한 경우) ─────────────

def _make_fake_task_json(prompt: str, persona: str, category: str) -> str:
    """도메인 무관 입력을 마치 태스크 목록처럼 잘못 처리한 rejected 응답."""
    fake_tasks_by_category = {
        "recipe": [
            ("재료 준비하기", 4, 3, 1, 1, "요리 시작 전 필수"),
            ("조리 순서 확인하기", 3, 2, 2, 1, "레시피 참고"),
            ("완성 후 담기", 2, 2, 1, 1, "플레이팅"),
        ],
        "math": [
            ("문제 읽기", 3, 2, 1, 1, "문제 파악"),
            ("풀이 과정 작성", 4, 4, 2, 1, "핵심 단계"),
            ("답 확인", 3, 3, 1, 1, "검산"),
        ],
        "default": [
            ("질문 분석하기", 3, 3, 2, 1, "내용 파악"),
            ("답변 준비하기", 4, 4, 3, 1, "핵심 내용"),
            ("결과 정리하기", 2, 2, 1, 1, "마무리"),
        ],
    }
    tasks_data = fake_tasks_by_category.get(category, fake_tasks_by_category["default"])
    tasks = [{"id": i + 1, "text": t[0]} for i, t in enumerate(tasks_data)]
    scores = [
        {
            "task_id": i + 1,
            "urgency": t[1],
            "importance": t[2],
            "dependency": t[3],
            "time_constraint": t[4],
            "reason": t[5],
        }
        for i, t in enumerate(tasks_data)
    ]
    fake_resp = {
        "tasks": tasks,
        "priority_order": [1, 2, 3],
        "scores": scores,
    }
    return json.dumps(fake_resp, ensure_ascii=False)


# ── 데이터 생성 ───────────────────────────────────────────────────────────────

def _make_prompt(raw_input: str, persona: str) -> str:
    return f"[{persona}의 오늘의 할 일 목록]\n{raw_input}"


def _make_chosen(category: str, reason_idx: int = 0) -> str:
    reasons = _REFUSAL_REASONS[category]
    reason = reasons[reason_idx % len(reasons)]
    resp = ScheduleResponse(tasks=[], priority_order=[], scores=[], refusal_reason=reason)
    return resp.model_dump_json()


def generate_sft_rows(limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    rng = random.Random(42)

    for (category, templates), persona in product(_PATTERNS, _PERSONAS):
        for i, template in enumerate(templates):
            prompt = _make_prompt(template, persona)
            chosen = _make_chosen(category, reason_idx=i)
            rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "persona": persona,
                "source": f"refusal_v2/{category}",
            })

    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows


def generate_dpo_rows(sft_rows: list[dict], dpo_limit: int | None = None) -> list[dict]:
    """SFT rows에서 DPO 쌍 생성 (chosen=올바른 거부, rejected=잘못된 태스크 처리)."""
    rng = random.Random(42)
    rows: list[dict] = []

    for row in sft_rows:
        category = row["source"].split("/")[-1]
        persona = row["persona"]
        prompt = row["prompt"]

        # 원본 입력 (prompt에서 헤더 제거)
        raw_input = prompt.split("\n", 1)[-1] if "\n" in prompt else prompt

        chosen = row["chosen"]
        rejected = _make_fake_task_json(raw_input, persona, category)

        rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "persona": persona,
            "source": row["source"],
        })

    rng.shuffle(rows)
    if dpo_limit:
        rows = rows[:dpo_limit]
    return rows


# ── 검증 ─────────────────────────────────────────────────────────────────────

def verify_rows(rows: list[dict], col: str = "chosen") -> tuple[int, int]:
    """parse_lenient 통과율 반환 (passed, total)."""
    passed = 0
    for row in rows:
        result = parse_lenient(row[col])
        if result is not None:
            passed += 1
        else:
            print(f"  [파싱 실패] {row[col][:80]}", file=sys.stderr)
    return passed, len(rows)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="거부 케이스 합성 (Phase B)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="SFT 행 수 상한 (dry-run: 20, 전체: None)"
    )
    parser.add_argument(
        "--dpo-limit", type=int, default=None,
        help="DPO 행 수 상한 (기본: SFT 행 수의 60%%)"
    )
    parser.add_argument(
        "--sft-out", default="data/refusals_sft_v2.parquet",
        help="SFT 출력 parquet 경로"
    )
    parser.add_argument(
        "--dpo-out", default="data/refusals_dpo_v2.parquet",
        help="DPO 출력 parquet 경로"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="생성 후 parse_lenient 통과율 검증"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="--limit 20 --verify 단축키"
    )
    args = parser.parse_args()

    if args.dry_run:
        args.limit = args.limit or 20
        args.verify = True

    print("[Phase B] 거부 케이스 합성")
    print(f"  패턴 수: {len(_PATTERNS)}종  ×  페르소나: {len(_PERSONAS)}개")
    print(f"  최대 생성 가능: {sum(len(t) for _, t in _PATTERNS) * len(_PERSONAS)}행")

    sft_rows = generate_sft_rows(limit=args.limit)
    dpo_limit = args.dpo_limit or int(len(sft_rows) * 0.6)
    dpo_rows = generate_dpo_rows(sft_rows, dpo_limit=dpo_limit)

    print(f"\n  SFT 행: {len(sft_rows)}개")
    print(f"  DPO 쌍: {len(dpo_rows)}개")

    if args.verify:
        print("\n[검증] chosen parse_lenient 통과율...")
        sft_passed, sft_total = verify_rows(sft_rows, "chosen")
        dpo_passed, dpo_total = verify_rows(dpo_rows, "chosen")
        dpo_rej_passed, dpo_rej_total = verify_rows(dpo_rows, "rejected")

        print(f"  SFT chosen  : {sft_passed}/{sft_total} ({sft_passed/sft_total*100:.1f}%)")
        print(f"  DPO chosen  : {dpo_passed}/{dpo_total} ({dpo_passed/dpo_total*100:.1f}%)")
        print(f"  DPO rejected: {dpo_rej_passed}/{dpo_rej_total} ({dpo_rej_passed/dpo_rej_total*100:.1f}%)")

        if sft_passed < sft_total or dpo_passed < dpo_total:
            print("\n[경고] parse 실패 행이 있습니다. 위 오류를 확인하세요.", file=sys.stderr)
            sys.exit(1)
        print("\n[OK] 전원 통과")

    if not args.dry_run:
        sft_path = Path(args.sft_out)
        dpo_path = Path(args.dpo_out)
        sft_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(sft_rows).to_parquet(str(sft_path), index=False)
        pd.DataFrame(dpo_rows).to_parquet(str(dpo_path), index=False)

        print("\n[저장]")
        print(f"  SFT: {sft_path}  ({len(sft_rows)}행)")
        print(f"  DPO: {dpo_path}  ({len(dpo_rows)}행)")
    else:
        print("\n[dry-run] 파일 저장 생략 (--dry-run 해제 시 저장)")
