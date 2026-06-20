#!/usr/bin/env python
"""v9 결정적 데이터셋 스캐폴드 생성 (직업 인지 · 다양성 · 실현가능 마감).

설계(체크포인트 #1 피드백 반영):
1. 마감 실현가능성: 스케줄을 먼저(우선순위 순 그리디) 잡고 마감 = 스케줄 종료 + tier 버퍼.
   → 모든 마감이 항상 충족 가능(모델에 마감위반을 학습시키지 않음).
2. 직업 부합: occupation 카테고리별 시나리오·도메인·체인테마 가중.
3. reasoning 정확성: 행마다 per-task facts(마감까지 시간/슬랙/마감충족/순위) 계산 → sonnet 주입.
4. 다양성(과적합 방지): 12개 생활 도메인 + 18종 체인테마를 순환 배정, 태스크 수·체인 길이·
   가용시간·현재시각(계절)·직업을 폭넓게 변주.

출력: outputs/v9/build/scaffold/batch_{NN}.json (배치별 행 리스트) + manifest.json

사용:
  uv run python scripts/v9/build_dataset.py --n 1600 --batch-size 16 --seed 20260619
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts/v9")
from timesorter.data import schema_v9 as S  # noqa: E402
from occupations import categorize, CAT_CONFIG, OFFICE_CATS  # noqa: E402

# ── 다양성 풀 ────────────────────────────────────────────────────────────────
DOMAINS = [
    "건강/병원", "금융/납부", "가족/돌봄", "학습/자기계발", "집안일/수리",
    "사교/약속", "행정/서류", "장보기/심부름", "디지털/예약", "운동/취미",
    "업무/사무", "고객/연락",
]
WORK_CHAINS = [
    "프로젝트 산출물 제작", "분기 실적 보고", "계약 검토·체결", "행사 기획·운영",
    "내부 감사 준비", "교육 자료 제작", "신제품 출시 준비", "예산안 편성",
]
PERSONAL_CHAINS = [
    "이사 준비", "국내·해외 여행 준비", "집들이 준비", "종합 건강검진 절차",
    "자격증 시험 준비", "공공 서류 발급", "김장·대량 요리", "연말정산 서류 준비",
    "결혼 준비", "중고 물품 정리·판매", "반려동물 입양 준비", "자녀 입학 준비",
]
RISK_CLAUSES = [
    "미처리 시 연체료 발생", "기한 초과 시 위약금 부과", "지연 시 고객 클레임 예상",
    "법정 신고 기한 — 연장 불가", "미응답 시 예약 자동 취소", "미제출 시 지원 자격 상실",
]

# 도메인별 세부 소재 풀 — 같은 도메인도 다른 제목이 나오도록 슬롯마다 순환 배정
MICRO_TOPICS = {
    "건강/병원": ["치과 스케일링", "독감 예방접종", "건강검진 결과 상담", "물리치료 세션",
                "안과 시력검사", "혈압약 처방 재발급", "피부과 진료", "정신건강 상담",
                "한방 침 치료", "도수치료 예약", "위내시경 예약", "백신 추가 접종"],
    "금융/납부": ["전기요금 납부", "신용카드 대금 결제", "자동차 보험 갱신", "적금 만기 처리",
                "관리비 정산", "통신비 요금제 변경", "세금 신고 자료 준비", "대출 이자 상환",
                "연금 납입 확인", "환전 신청", "주식 배당금 확인", "보험금 청구"],
    "가족/돌봄": ["부모님 병원 동행", "아이 학원 등록", "반려견 미용 예약", "가족 생일상 준비",
                "조부모 안부 방문", "아이 예방접종 동행", "배우자 기념일 선물", "가족 여행 일정 조율",
                "어린이집 상담", "노부모 약 정리"],
    "학습/자기계발": ["토익 모의고사 풀이", "온라인 강의 수강", "독서 모임 발제 준비", "자격증 기출 복습",
                  "코딩 연습 문제", "외국어 회화 연습", "세미나 자료 정독", "글쓰기 연습",
                  "유튜브 강의 정리", "스터디 과제"],
    "집안일/수리": ["보일러 점검 예약", "욕실 실리콘 보수", "에어컨 필터 청소", "베란다 정리",
                "전구 교체", "세탁기 청소", "가구 조립", "벽면 균열 보수", "방충망 수리", "냉장고 정리"],
    "사교/약속": ["친구와 저녁 약속", "동호회 모임 참석", "동창회 일정 조율", "지인 결혼식 참석",
                "이웃 집들이 방문", "온라인 모임 호스팅", "멘토 커피챗", "봉사활동 참여"],
    "행정/서류": ["인감증명 발급", "주민등록등본 발급", "여권 갱신 신청", "전입신고",
                "자동차 등록 갱신", "각종 증명서 제출", "민원 신청서 작성", "비자 서류 준비"],
    "장보기/심부름": ["주간 장보기", "택배 반품 접수", "세탁소 옷 찾기", "약국 처방약 수령",
                  "선물 포장 의뢰", "우체국 등기 발송", "마트 생필품 구매", "꽃집 화분 픽업"],
    "디지털/예약": ["식당 예약", "공연 티켓 예매", "미용실 예약", "항공편 예약",
                "숙소 예약", "비대면 진료 예약", "온라인 강좌 등록", "구독 서비스 정리"],
    "운동/취미": ["헬스장 PT 세션", "주말 등산 계획", "수영 강습", "악기 연습",
                "사진 출사", "요가 클래스", "자전거 정비", "베이킹 도전"],
    "업무/사무": ["주간 업무 보고 정리", "거래처 미팅 준비", "경비 영수증 처리", "회의록 작성",
                "이메일 회신 정리", "결재 문서 검토", "고객 견적서 작성", "재고 현황 점검"],
    "고객/연락": ["고객 문의 회신", "거래처 일정 조율", "신규 리드 팔로업", "클레임 응대",
                "계약 갱신 안내", "납품 일정 확인 연락", "협력사 미팅 요청", "AS 접수 처리"],
    "생활": ["집 정리정돈", "냉장고 식재료 점검", "주간 식단 계획", "분리수거"],
}

# 체인 테마별 구체 변형 — 같은 테마도 다른 제목 나오게
CHAIN_SUBJECTS = {
    "프로젝트 산출물 제작": ["신규 앱 기획서", "데이터 분석 리포트", "마케팅 캠페인안", "UX 개선 제안서"],
    "분기 실적 보고": ["2분기 매출 보고", "팀 성과 대시보드", "비용 절감 실적", "지역별 판매 분석"],
    "계약 검토·체결": ["공급사 신규 계약", "임대차 계약", "외주 용역 계약", "연간 유지보수 계약"],
    "행사 기획·운영": ["사내 워크숍", "고객 초청 세미나", "신제품 런칭 행사", "창립기념 행사"],
    "내부 감사 준비": ["회계 감사 대응", "보안 점검 대응", "품질 인증 심사", "안전 점검 대비"],
    "교육 자료 제작": ["신입 온보딩 가이드", "안전교육 교안", "제품 매뉴얼", "고객 응대 매뉴얼"],
    "신제품 출시 준비": ["베타 출시 점검", "패키지 디자인 확정", "출시 보도자료", "사전예약 페이지"],
    "예산안 편성": ["내년도 부서 예산", "프로젝트 예산안", "마케팅 예산 배분", "설비 투자 계획"],
    "이사 준비": ["원룸에서 투룸으로 이사", "사무실 이전", "부모님 댁 이사", "타지역 전근 이사"],
    "국내·해외 여행 준비": ["제주 가족여행", "일본 자유여행", "부산 출장 겸 여행", "유럽 배낭여행"],
    "집들이 준비": ["신혼집 집들이", "이사 후 집들이", "친구 초대 홈파티", "부모님 초대 식사"],
    "종합 건강검진 절차": ["국가 건강검진", "종합 정밀검진", "내시경 포함 검진", "심혈관 정밀검사"],
    "자격증 시험 준비": ["정보처리기사 시험", "공인중개사 시험", "토익 스피킹", "한식조리기능사"],
    "공공 서류 발급": ["부동산 등기 서류", "사업자등록 서류", "장학금 신청 서류", "복지 지원 서류"],
    "김장·대량 요리": ["겨울 김장", "명절 음식 준비", "반찬 대량 조리", "행사용 도시락"],
    "연말정산 서류 준비": ["연말정산 자료", "종합소득세 신고", "의료비 영수증 정리", "기부금 영수증"],
    "결혼 준비": ["예식장 계약", "신혼여행 준비", "청첩장 제작", "혼수 준비"],
    "중고 물품 정리·판매": ["가전 중고 판매", "옷장 정리 판매", "이사 전 물품 정리", "도서 중고 처분"],
    "반려동물 입양 준비": ["강아지 입양 준비", "고양이 용품 마련", "반려동물 병원 등록", "펫 보험 가입"],
    "자녀 입학 준비": ["초등 입학 준비", "어린이집 입소", "중학교 배정 준비", "대학 입학 서류"],
}
_GENERIC_MODIFIERS = ["", "", "", ""]

# tier → (deadline_proximity 기준, urgency 기본, 버퍼 분 범위)
TIER_TIGHT = "tight_today"   # 오늘 빠듯
TIER_TODAY = "today"          # 오늘 여유
TIER_SOON = "soon"            # 1-2일
TIER_LATER = "later"          # 3-7일
TIER_NONE = "none"            # 마감 없음

_TIER_BUFFER = {  # 스케줄 종료 후 마감까지 버퍼(분) 범위 — 항상 ≥0 ⇒ 실현가능
    TIER_TIGHT: (10, 40), TIER_TODAY: (90, 240),
    TIER_SOON: (1440, 2880), TIER_LATER: (4320, 10080),
}
_TIER_RANKKEY = {TIER_TIGHT: 0, TIER_TODAY: 1, TIER_SOON: 2, TIER_LATER: 3, TIER_NONE: 4}
_DUR_CHOICES = [15, 30, 30, 45, 60, 60, 90, 120]


def _avail_start(avail: str) -> dt.time:
    return S._parse_window(avail)[0]


def gen_row(persona: dict, date: dt.date, rng: random.Random,
            dctr: list[int], mctr: dict[str, int]) -> dict:
    occs = persona["persona"]["occupations"]
    cat = persona.get("_meta", {}).get("category") or categorize(" ".join(occs))
    cfg = CAT_CONFIG.get(cat, CAT_CONFIG["general"])
    core_domains, work_ratio = cfg["domains"], cfg["work_ratio"]
    avail = persona["persona"]["availability"]
    start_t = _avail_start(avail)
    current_time = dt.datetime.combine(date, start_t, tzinfo=S.KST)

    # ── 구성: 체인 0-2개 + 독립 태스크 → 총 4-8개 ──
    n_chains = rng.choices([0, 1, 1, 2], weights=[2, 4, 4, 2])[0]
    specs: list[dict] = []
    gid = 0
    used_chain_themes = []
    for _ in range(n_chains):
        gid += 1
        is_work = rng.random() < work_ratio
        pool = WORK_CHAINS if is_work else PERSONAL_CHAINS
        theme = rng.choice([t for t in pool if t not in used_chain_themes] or pool)
        used_chain_themes.append(theme)
        clen = rng.choice([3, 3, 4, 4, 5]) if n_chains == 1 else rng.choice([3, 3, 4])
        # 체인 최종 tier: 오늘 또는 1-2일
        fin_tier = rng.choices([TIER_TIGHT, TIER_TODAY, TIER_SOON],
                               weights=[2, 4, 3])[0]
        subject = rng.choice(CHAIN_SUBJECTS.get(theme, [theme]))  # 구체 주제 변형
        for pos in range(1, clen + 1):
            specs.append({
                "kind": "chain", "chain_group": gid, "chain_pos": pos,
                "is_final_chain": pos == clen, "risk": False, "risk_clause": "",
                "tier": fin_tier if pos == clen else TIER_NONE,  # 선행 단계는 자체 마감 없음
                "chain_theme": theme, "chain_subject": subject,
                "domain_hint": "업무/사무" if is_work else "생활", "micro_topic": "",
            })
    n_indep = rng.randint(2, 5) if n_chains else rng.randint(4, 7)
    for _ in range(n_indep):
        # 도메인 순환 배정(전역 카운터) + 카테고리 코어 도메인 섞기
        if rng.random() < 0.55:
            dom = core_domains[dctr[0] % len(core_domains)]
        else:
            dom = DOMAINS[dctr[0] % len(DOMAINS)]
        dctr[0] += 1
        # 도메인별 micro-topic 순환 → 같은 도메인도 다른 소재
        pool = MICRO_TOPICS.get(dom, MICRO_TOPICS["생활"])
        mi = mctr.get(dom, rng.randrange(len(pool)))
        micro = pool[mi % len(pool)]
        mctr[dom] = mi + 1
        tier = rng.choices([TIER_TIGHT, TIER_TODAY, TIER_SOON, TIER_LATER, TIER_NONE],
                           weights=[2, 4, 3, 2, 3])[0]
        # risk(에스컬레이션·위약금)는 사무 직군의 업무 태스크에만 — 개인 일정에 부적합 방지
        risk = (rng.random() < 0.18 and tier in (TIER_TIGHT, TIER_TODAY, TIER_SOON)
                and cat in OFFICE_CATS and dom in ("업무/사무", "고객/연락"))
        specs.append({
            "kind": "today" if tier in (TIER_TIGHT, TIER_TODAY) else (
                "future" if tier in (TIER_SOON, TIER_LATER) else "none"),
            "chain_group": 0, "chain_pos": 0, "is_final_chain": False,
            "risk": risk, "risk_clause": rng.choice(RISK_CLAUSES) if risk else "",
            "tier": tier, "chain_theme": "", "chain_subject": "",
            "domain_hint": dom, "micro_topic": micro,
        })

    # 블록 단위 배치: 체인은 chain_pos 순서로 묶어 **연속 task_id** 부여(haiku 단계 혼동 제거).
    # 독립 태스크는 개별 블록. 블록 순서만 섞고 체인 내부 순서는 보존.
    cb: dict[int, list] = {}
    indep = []
    for s in specs:
        if s["chain_group"]:
            cb.setdefault(s["chain_group"], []).append(s)
        else:
            indep.append(s)
    blocks_list = [sorted(v, key=lambda x: x["chain_pos"]) for v in cb.values()] + [[s] for s in indep]
    rng.shuffle(blocks_list)
    specs = [s for blk in blocks_list for s in blk]
    for i, s in enumerate(specs, 1):
        s["idx"] = i
        s["task_id"] = f"task_{i:03d}"
        s["estimated_duration_minutes"] = rng.choice(_DUR_CHOICES)

    # ── 우선순위(rank): 체인=블록, tier 우선 + risk 가산 ──
    chains: dict[int, list] = {}
    for s in specs:
        if s["chain_group"]:
            chains.setdefault(s["chain_group"], []).append(s)
    for members in chains.values():
        members.sort(key=lambda s: s["chain_pos"])
    chain_pairs = []
    for members in chains.values():
        for a, b in zip(members, members[1:]):
            chain_pairs.append((a["task_id"], b["task_id"]))

    def block_key(spec_or_members):
        # 더 급한 tier(작은 수) 우선, risk 우선
        if isinstance(spec_or_members, list):
            tier = min(_TIER_RANKKEY[m["tier"]] for m in spec_or_members)
            risk = any(m["risk"] for m in spec_or_members)
        else:
            tier = _TIER_RANKKEY[spec_or_members["tier"]]
            risk = spec_or_members["risk"]
        return (tier, 0 if risk else 1)

    blocks = []
    used = set()
    for members in chains.values():
        blocks.append((block_key(members), [m["task_id"] for m in members]))
        used.update(m["task_id"] for m in members)
    for s in specs:
        if s["task_id"] not in used:
            blocks.append((block_key(s), [s["task_id"]]))
    blocks.sort(key=lambda b: b[0])
    ranked_ids = [tid for _, ids in blocks for tid in ids]
    rank = {tid: i for i, tid in enumerate(ranked_ids, 1)}

    # ── 스케줄(우선순위 순 그리디) → 실현가능 ──
    dur_map = {s["task_id"]: s["estimated_duration_minutes"] for s in specs}
    sched = S.build_schedule(current_time, avail,
                             [(tid, dur_map[tid]) for tid in ranked_ids])

    # ── 마감 = 스케줄 종료 + tier 버퍼 (체인은 최종에만) ──
    iso_dl = {}
    for s in specs:
        tid = s["task_id"]
        tier = s["tier"]
        if tier == TIER_NONE:
            iso_dl[tid] = None
            continue
        _, end = sched[tid]
        lo, hi = _TIER_BUFFER[tier]
        dl = end + dt.timedelta(minutes=rng.randint(lo, hi))
        iso_dl[tid] = dl.isoformat()

    # 체인 멤버 effective deadline(최종) + 잔여작업량
    eff_dl, remaining = {}, {}
    for members in chains.values():
        ids = [m["task_id"] for m in members]
        fin = iso_dl[ids[-1]]
        suf = 0
        for tid in reversed(ids):
            suf += dur_map[tid]
            remaining[tid] = suf
            eff_dl[tid] = fin

    # ── 점수(실제 마감 기준) + facts ──
    scoring, facts = {}, {}
    for s in specs:
        tid = s["task_id"]
        in_chain = bool(s["chain_group"])
        eff = eff_dl.get(tid, iso_dl[tid])
        eff_d = S.parse_dt(eff) if eff else None
        slack_dur = remaining.get(tid, dur_map[tid])
        dp = S.deadline_proximity_score(current_time, eff_d)
        ur = S.urgency_score(current_time, eff_d, slack_dur, s["risk"])
        imp = S.importance_score(s["kind"], s["risk"], in_chain)
        ch = S.chaining_score(in_chain, s["is_final_chain"])
        total = S.compute_total_score(dp, imp, ch, ur)
        scoring[tid] = {"deadline_proximity": dp, "task_importance": imp,
                        "task_chaining": ch, "urgency": ur, "total_score": total}
        st, en = sched[tid]
        own_dl = iso_dl[tid]
        h2d = round((S.parse_dt(own_dl) - current_time).total_seconds() / 3600, 1) if own_dl else None
        facts[tid] = {
            "rank": rank[tid], "deadline": own_dl,
            "hours_to_deadline": h2d,
            "scheduled": f"{st.strftime('%H:%M')}~{en.strftime('%H:%M')}",
            "deadline_met": True,
            "in_chain": in_chain, "chain_theme": s["chain_theme"],
            "domain": s["domain_hint"], "risk_clause": s["risk_clause"],
        }

    slots = [{
        "task_id": s["task_id"], "kind": s["kind"], "deadline": iso_dl[s["task_id"]],
        "estimated_duration_minutes": dur_map[s["task_id"]],
        "in_chain": bool(s["chain_group"]), "chain_group": s["chain_group"],
        "chain_pos": s["chain_pos"], "is_final_chain": s["is_final_chain"],
        "risk": s["risk"], "risk_clause": s["risk_clause"],
        "chain_theme": s["chain_theme"], "chain_subject": s.get("chain_subject", ""),
        "domain_hint": s["domain_hint"], "micro_topic": s.get("micro_topic", ""),
    } for s in specs]

    return {
        "persona_id": persona["persona_id"], "persona": {k: v for k, v in persona["persona"].items()},
        "occ_category": cat, "current_time": current_time.isoformat(),
        "slots": slots, "scoring": scoring, "priority_rank": rank,
        "schedule": {tid: {"start_time": S.iso(st), "end_time": S.iso(en)}
                     for tid, (st, en) in sched.items()},
        "chain_pairs": chain_pairs, "facts": facts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--personas", default="data/v9/personas.json")
    ap.add_argument("--out-dir", default="outputs/v9/build")
    args = ap.parse_args()

    personas = json.loads(Path(args.personas).read_text())
    rng = random.Random(args.seed)
    dctr = [rng.randint(0, 11)]
    mctr: dict[str, int] = {}
    rows = []
    for i in range(args.n):
        persona = personas[rng.randrange(len(personas))]
        # 현재시각(계절) 변주: 2026 전역
        date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 364))
        rows.append(gen_row(persona, date, rng, dctr, mctr))

    out = Path(args.out_dir)
    (out / "scaffold").mkdir(parents=True, exist_ok=True)
    bs = args.batch_size
    n_batch = (len(rows) + bs - 1) // bs
    for b in range(n_batch):
        chunk = rows[b * bs:(b + 1) * bs]
        for j, r in enumerate(chunk):
            r["row_id"] = b * bs + j
        (out / "scaffold" / f"batch_{b:03d}.json").write_text(
            json.dumps(chunk, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps(
        {"n": len(rows), "n_batch": n_batch, "batch_size": bs, "seed": args.seed},
        ensure_ascii=False, indent=2))

    # 진단
    from collections import Counter
    cats = Counter(r["occ_category"] for r in rows)
    nt = Counter(len(r["slots"]) for r in rows)
    nch = Counter(len(set(s["chain_group"] for s in r["slots"] if s["chain_group"])) for r in rows)
    themes = Counter(s["chain_theme"] for r in rows for s in r["slots"] if s["chain_theme"])
    print(f"[saved] {out}/scaffold/ — {len(rows)}행 / {n_batch}배치")
    print("occ_category:", dict(cats))
    print("태스크수 분포:", dict(sorted(nt.items())))
    print("체인수 분포:", dict(sorted(nch.items())))
    print("체인테마 종수:", len(themes), "| 상위:", themes.most_common(5))


if __name__ == "__main__":
    main()
