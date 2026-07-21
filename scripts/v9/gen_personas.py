#!/usr/bin/env python
"""v9 페르소나 생성 — Nemotron-Personas-Korea → 구조화 페르소나 카드 10-30개.

이름은 제거(사용자별 익명 페르소나). occupation 다양성·연령·성별 균형을 맞춰 샘플링.
출력: data/v9/personas.json (UserPersona 스키마 호환 딕셔너리 리스트 + 원본 메타).

사용:
  uv run python scripts/v9/gen_personas.py --n 20 --seed 9
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, "scripts/v9")
from occupations import categorize, CATEGORY_NAMES  # noqa: E402

NEMOTRON = Path("data/nemotron_personas_korea.parquet")
OUT = Path("data/v9/personas.json")

# 직업군 → 가용 시간창 (없으면 기본). 키는 occupation 부분 매칭.
_AVAIL = [
    (["경비", "시설", "보안"], "06:00-22:00"),
    (["조리", "주방", "요리", "한식"], "10:00-22:00"),
    (["청소", "환경"], "07:00-19:00"),
    (["운전", "배달", "택배", "기사"], "08:00-20:00"),
    (["프리랜서", "작가", "디자이", "개발"], "10:00-23:00"),
    (["상담", "전화", "콜"], "09:00-18:00"),
    (["사무", "회계", "경리", "비서", "관리", "영업", "마케팅"], "09:00-18:00"),
    (["무직", "구직", "은퇴", "주부"], "08:00-22:00"),
]
_DEFAULT_AVAIL = "09:00-18:00"


def _name_of(persona: str) -> str:
    m = re.match(r"^(\S{2,4})\s*씨", persona.strip())
    return m.group(1) if m else ""


_JOSA = {"는": "은", "가": "이", "를": "을", "와": "과", "랑": "이랑"}


def _strip_name(text: str, name: str) -> str:
    if not text:
        return ""
    if name:
        # "{name} 씨<조사>" → "본인<교정조사>" (받침 있는 '본인'에 맞게 조사 교정)
        for sp in (f"{name} 씨", f"{name}씨"):
            for josa, fixed in _JOSA.items():
                text = text.replace(sp + josa, "본인" + fixed)
            text = text.replace(sp, "본인")
        text = text.replace(name, "본인")
    return re.sub(r"\s+", " ", text).strip()


def _availability(occ: str) -> str:
    for keys, win in _AVAIL:
        if any(k in occ for k in keys):
            return win
    return _DEFAULT_AVAIL


def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?다])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def to_card(row: pd.Series) -> dict:
    persona = str(row.get("persona", ""))
    name = _name_of(persona)
    occ = str(row.get("occupation", "")).strip()
    prof = str(row.get("professional_persona", ""))
    goals = str(row.get("career_goals_and_ambitions", ""))
    gender = {"남자": "male", "여자": "female"}.get(str(row.get("sex", "")).strip(), "")
    district = str(row.get("district", "")).replace("-", " ").strip()
    bio = _strip_name(_first_sentences(persona, 2), name)
    status = _strip_name(_first_sentences(prof, 1) + " " + _first_sentences(goals, 1), name)
    return {
        "persona": {
            "occupations": [occ] if occ and occ != "무직" else (["무직(구직 중)"] if occ == "무직" else []),
            "detailed_status": status[:300],
            "age": int(row.get("age", 0)),
            "gender": gender,
            "location": {"country": "South Korea", "city": district},
            "bio": bio[:300],
            "availability": _availability(occ),
        },
        "_meta": {"uuid": str(row.get("uuid", "")), "occupation": occ,
                  "marital_status": str(row.get("marital_status", "")),
                  "education_level": str(row.get("education_level", ""))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=44, help="페르소나 수 (12 직업군 균등)")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--out", default=str(OUT), help="출력 경로")
    ap.add_argument("--exclude", default="", help="제외할 uuid가 든 personas json(쉼표 구분) — held-out용")
    args = ap.parse_args()

    df = pd.read_parquet(NEMOTRON, columns=[
        "uuid", "persona", "professional_persona", "career_goals_and_ambitions",
        "sex", "age", "occupation", "district", "marital_status", "education_level"])
    df = df[df["age"].between(19, 75)].copy()
    excl = set()
    for p in [x for x in args.exclude.split(",") if x.strip()]:
        for c in json.loads(Path(p).read_text()):
            u = c.get("_meta", {}).get("uuid")
            if u:
                excl.add(u)
    if excl:
        before = len(df)
        df = df[~df["uuid"].isin(excl)].copy()
        print(f"[held-out] 제외 uuid {len(excl)}개 → {before}→{len(df)}행")

    # 직업→직업군 매핑 (고유 직업 2,120개만 분류 → 빠름)
    occ2cat = {o: categorize(o) for o in df["occupation"].dropna().unique()}
    df["cat"] = df["occupation"].map(occ2cat)

    # 직업군별 균등 배분 (나머지는 앞 직업군부터)
    cats = CATEGORY_NAMES
    base, rem = divmod(args.n, len(cats))
    quota = {c: base + (1 if i < rem else 0) for i, c in enumerate(cats)}

    picks = []
    for ci, cat in enumerate(cats):
        sub = df[df["cat"] == cat]
        k = quota[cat]
        if len(sub) == 0 or k == 0:
            continue
        # 직업 중복 최소화: 직업군 내 서로 다른 직업을 우선 선택, 직업당 1명씩
        occ_pool = list(sub["occupation"].dropna().unique())
        rng = random.Random(args.seed + ci)
        rng.shuffle(occ_pool)
        rows_idx = []
        for occ in occ_pool[:k]:
            cand = sub[sub["occupation"] == occ]
            rows_idx.append(cand.sample(1, random_state=args.seed + ci).index[0])
        if len(rows_idx) < k:  # 직업 종류 부족 시 보충
            extra = sub.drop(rows_idx).sample(
                min(k - len(rows_idx), len(sub) - len(rows_idx)),
                random_state=args.seed + 100 + ci)
            rows_idx += list(extra.index)
        for idx in rows_idx:
            picks.append((cat, df.loc[idx]))

    cards = []
    for i, (cat, row) in enumerate(picks, 1):
        c = to_card(row)
        c["persona_id"] = f"p{i:02d}"
        c["_meta"]["category"] = cat
        cards.append(c)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path} — {len(cards)} personas (목표 {args.n})")
    from collections import Counter
    catc = Counter(c["_meta"]["category"] for c in cards)
    print("직업군 분포:", dict(catc))
    print("고유 직업 수:", len({c["_meta"]["occupation"] for c in cards}))
    print("연령:", sorted(c["persona"]["age"] for c in cards))


if __name__ == "__main__":
    main()
