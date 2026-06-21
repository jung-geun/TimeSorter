#!/usr/bin/env python
"""v9 EN-US 페르소나 생성 — Nemotron-Personas-USA → 구조화 페르소나 카드.

KR `gen_personas.py`의 미국판. 차이: 영어 이름 익명화("the user", 조사 없음),
occupation snake_case humanize, geo=city+state, sex Female/Male→female/male,
country="United States". 직업군 균등 샘플링(occupations_en).

사용:
  uv run python scripts/v9/gen_personas_en.py --n 48 --seed 9
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts/v9")
from occupations_en import categorize, CATEGORY_NAMES  # noqa: E402
from content_en import COUNTRY  # noqa: E402

NEMOTRON = Path("data/nemotron_personas_usa_shard0.parquet")
OUT = Path("data/v9/personas_en.json")

# 직업군 키워드 → 가용 시간창 (미국 근무 패턴). 없으면 기본.
_AVAIL = [
    (["security", "guard", "firefighter", "police", "patrol"], "06:00-22:00"),
    (["cook", "chef", "food", "server", "waiter", "bartender", "barista"], "10:00-22:00"),
    (["janitor", "cleaner", "housekeep", "maid"], "07:00-19:00"),
    (["driver", "delivery", "truck", "courier"], "07:00-20:00"),
    (["nurse", "nursing", "paramedic", "emt"], "07:00-19:00"),
    (["develop", "engineer", "designer", "writer", "artist", "research"], "10:00-23:00"),
    (["retail", "sales", "cashier", "store", "merchand"], "10:00-20:00"),
    (["secretary", "administrative", "office", "manager", "analyst", "accountant",
      "clerk", "consultant"], "09:00-17:00"),
    (["not_in_workforce", "no_occupation", "retired", "homemaker", "student"], "08:00-22:00"),
]
_DEFAULT_AVAIL = "09:00-17:00"

_NAME_RE = re.compile(r"^([A-Z][a-z'’.\-]+(?:\s+[A-Z][a-z'’.\-]+){0,2})")


def _name_of(persona: str) -> tuple[str, str]:
    """페르소나 첫 토큰들에서 (full name, first name) 추출."""
    m = _NAME_RE.match(persona.strip())
    if not m:
        return "", ""
    full = m.group(1).strip()
    return full, full.split()[0]


def _cap_sentences(text: str) -> str:
    """문장 시작 글자 대문자화(맨 앞 + .!? 뒤). 치환된 'the user'가 문두면 'The user'로."""
    if not text:
        return text
    text = re.sub(r"^(\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"([.!?]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)


def _clip(text: str, n: int = 300) -> str:
    """단어/문장 경계에서 절단(단어 중간 컷 방지)."""
    if len(text) <= n:
        return text
    cut = text[:n]
    for p in (". ", "! ", "? "):
        idx = cut.rfind(p)
        if idx >= n * 0.5:
            return cut[:idx + 1].strip()
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut).strip()


def _strip_name(text: str, full: str, first: str) -> str:
    """이름을 'the user'로 익명화(영어는 조사 없음) + 문장 첫 글자 대문자 보정."""
    if not text:
        return ""
    if full:
        # 소유격 우선 치환 후 일반
        text = re.sub(rf"\b{re.escape(full)}'s\b", "the user's", text)
        text = re.sub(rf"\b{re.escape(full)}\b", "the user", text)
    if first:
        text = re.sub(rf"\b{re.escape(first)}'s\b", "the user's", text)
        text = re.sub(rf"\b{re.escape(first)}\b", "the user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _cap_sentences(text)


def _humanize_occ(occ: str) -> str:
    """snake_case occupation code → 표시용 ('software_developer' → 'Software Developer')."""
    return " ".join(w.capitalize() for w in occ.replace("_", " ").split())


def _availability(occ: str) -> str:
    o = occ.lower()
    for keys, win in _AVAIL:
        if any(k in o for k in keys):
            return win
    return _DEFAULT_AVAIL


def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def to_card(row: pd.Series) -> dict:
    persona = str(row.get("persona", ""))
    full, first = _name_of(persona)
    occ = str(row.get("occupation", "")).strip()
    prof = str(row.get("professional_persona", ""))
    goals = str(row.get("career_goals_and_ambitions", ""))
    gender = {"Female": "female", "Male": "male"}.get(str(row.get("sex", "")).strip(), "")
    city = str(row.get("city", "")).strip()
    state = str(row.get("state", "")).strip()
    loc_city = f"{city}, {state}" if city and state else (city or state)
    bio = _clip(_strip_name(_first_sentences(persona, 2), full, first))
    status = _clip(_strip_name(_first_sentences(prof, 1) + " " + _first_sentences(goals, 1), full, first))
    if occ == "not_in_workforce":
        occupations = ["Not currently employed (job seeking)"]
    elif occ in ("no_occupation", ""):
        occupations = []
    else:
        occupations = [_humanize_occ(occ)]
    return {
        "persona": {
            "occupations": occupations,
            "detailed_status": status,
            "age": int(row.get("age", 0) or 0),
            "gender": gender,
            "location": {"country": COUNTRY, "city": loc_city},
            "bio": bio,
            "availability": _availability(occ),
        },
        "_meta": {"uuid": str(row.get("uuid", "")), "occupation": occ,
                  "marital_status": str(row.get("marital_status", "")),
                  "education_level": str(row.get("education_level", ""))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="페르소나 수 (12 직업군 균등)")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--out", default=str(OUT), help="출력 경로")
    ap.add_argument("--exclude", default="", help="제외할 uuid가 든 personas json(쉼표 구분) — held-out용")
    args = ap.parse_args()

    df = pd.read_parquet(NEMOTRON, columns=[
        "uuid", "persona", "professional_persona", "career_goals_and_ambitions",
        "sex", "age", "occupation", "city", "state", "marital_status", "education_level"])
    df = df[df["age"].between(19, 70)].copy()
    # held-out: 학습 페르소나 uuid 제외(누출 방지)
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

    occ2cat = {o: categorize(o) for o in df["occupation"].dropna().unique()}
    df["cat"] = df["occupation"].map(occ2cat)

    cats = CATEGORY_NAMES
    base, rem = divmod(args.n, len(cats))
    quota = {c: base + (1 if i < rem else 0) for i, c in enumerate(cats)}

    picks = []
    for ci, cat in enumerate(cats):
        sub = df[df["cat"] == cat]
        k = quota[cat]
        if len(sub) == 0 or k == 0:
            continue
        occ_pool = list(sub["occupation"].dropna().unique())
        rng = random.Random(args.seed + ci)
        rng.shuffle(occ_pool)
        rows_idx = []
        for occ in occ_pool[:k]:
            cand = sub[sub["occupation"] == occ]
            rows_idx.append(cand.sample(1, random_state=args.seed + ci).index[0])
        if len(rows_idx) < k:
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
