"""v9 직업군 정의 — Nemotron 2,120개 직업을 12개 그룹으로 분류 + 그룹별 생성 설정.

gen_personas.py(균등 샘플링)·build_dataset.py(분류+시나리오 설정) 공용.
순서 중요: 구체 패턴을 먼저 매칭(office_admin 같은 광범위 키워드는 뒤로).
"""
from __future__ import annotations

import re

# (category, 정규식) — 위에서부터 첫 매칭. 마지막 general은 fallback.
# 구직중/전직/무직은 최우선으로 jobseeker_senior에 라우팅(활성 직업군과 분리 → 업무 부정합 방지).
CATEGORIES: list[tuple[str, str]] = [
    ("jobseeker_senior", r"구직중|무직|은퇴|퇴직|주부|전직"),
    ("healthcare", r"간호|의무|약사|의사|요양|보건|치료|물리치료|방사선|임상|치위생|간병|위생사|영양사"),
    ("education", r"보육|유치원|교사|강사|교수|학원|교육|훈련|교직"),
    ("security_safety", r"경비|보안|안전|경호|소방|순찰|방재|경찰"),
    ("transport_logistics", r"운전|배달|기사|화물|지게차|철도|운송|택배|물류|선박|항해|항공|버스|택시"),
    ("service_food", r"조리|주방|음식|요리|한식|양식|제과|제빵|바리스타|카페|서빙|미용|이용|숙박|승무|접객|세탁"),
    ("sales_retail", r"영업|판매|상점|쇼핑|점원|매장|노점|중개|판촉|계산원|상품"),
    ("manual_production", r"청소|하역|적재|생산|기능|단순|제조|가공|조작|용접|도장|환경|설비|건설|배관|전기공|목공|포장"),
    ("professional_tech", r"컨설턴트|전문가|연구|조사|법률|변호사|회계사|세무사|기술자|엔지니어|개발|프로그래|디자이|건축|설계|분석|통역|번역|기자|작가"),
    ("management", r"임원|경영자|관리자|대표|사장|소규모 ?상점 ?경영|경영 ?기획|점주"),
    ("office_admin", r"사무|경리|회계|비서|기획|총무|문서|행정|인사|상담|접수|데이터"),
]
_GENERAL = "general"

# 그룹별: core 도메인(독립 태스크 가중), work_ratio(체인이 사무성 업무일 확률).
# WORK_CHAINS(보고서·예산·계약 등)는 사무성이므로 화이트칼라 직군만 사용 → work_ratio>0.
# 현장직/서비스/의료/구직/일반은 work_ratio=0 → 개인 체인만(직업 부합 결함 제거).
# 독립 도메인에서 현장직의 "업무/사무"는 빼고 직무 친화 도메인으로 대체.
OFFICE_CATS = {"office_admin", "professional_tech", "management", "sales_retail", "education"}
CAT_CONFIG: dict[str, dict] = {
    "healthcare":          {"domains": ["건강/병원", "가족/돌봄", "장보기/심부름", "학습/자기계발"], "work_ratio": 0.0},
    "education":           {"domains": ["학습/자기계발", "가족/돌봄", "업무/사무", "사교/약속"], "work_ratio": 0.40},
    "security_safety":     {"domains": ["집안일/수리", "건강/병원", "장보기/심부름", "운동/취미"], "work_ratio": 0.0},
    "transport_logistics": {"domains": ["장보기/심부름", "집안일/수리", "건강/병원", "가족/돌봄"], "work_ratio": 0.0},
    "service_food":        {"domains": ["장보기/심부름", "가족/돌봄", "집안일/수리", "건강/병원"], "work_ratio": 0.0},
    "sales_retail":        {"domains": ["고객/연락", "업무/사무", "디지털/예약", "금융/납부"], "work_ratio": 0.50},
    "manual_production":   {"domains": ["집안일/수리", "장보기/심부름", "건강/병원", "가족/돌봄"], "work_ratio": 0.0},
    "professional_tech":   {"domains": ["업무/사무", "학습/자기계발", "고객/연락", "디지털/예약"], "work_ratio": 0.55},
    "management":          {"domains": ["업무/사무", "고객/연락", "금융/납부", "사교/약속"], "work_ratio": 0.60},
    "office_admin":        {"domains": ["업무/사무", "고객/연락", "디지털/예약", "학습/자기계발"], "work_ratio": 0.55},
    "jobseeker_senior":    {"domains": ["학습/자기계발", "행정/서류", "건강/병원", "가족/돌봄"], "work_ratio": 0.0},
    _GENERAL:              {"domains": ["가족/돌봄", "건강/병원", "집안일/수리", "사교/약속"], "work_ratio": 0.0},
}

_COMPILED = [(c, re.compile(p)) for c, p in CATEGORIES]


def categorize(occ: str) -> str:
    for cat, pat in _COMPILED:
        if pat.search(occ):
            return cat
    return _GENERAL


CATEGORY_NAMES = [c for c, _ in CATEGORIES] + [_GENERAL]
