"""v9 EN-US 직업군 정의 — Nemotron-Personas-USA의 snake_case occupation 코드를 12개 그룹으로
분류 + 그룹별 생성 설정. KR `occupations.py`의 미국판 (동일 카테고리 키 → content_en 도메인 정합).

USA occupation은 BLS/Census 계열 snake_case 통제어휘(software_developer, registered_nurse,
construction_laborer …)라 한국어 자유텍스트보다 분류가 안정적. 순서 중요(구체 먼저, jobseeker 최우선).
gen_personas_en.py·build_dataset_en.py 공용.
"""
from __future__ import annotations

import re

# (category, 키워드 정규식) — 위에서부터 첫 매칭. snake_case 토큰 부분일치.
# not_in_workforce/no_occupation 등 비근로는 최우선으로 jobseeker_senior에 라우팅.
CATEGORIES: list[tuple[str, str]] = [
    ("jobseeker_senior", r"not_in_workforce|no_occupation|unemployed|retired|homemaker|student"),
    ("healthcare", r"nurs|physician|doctor|medical|health|therap|dental|dentist|pharmac|"
                   r"surgeon|veterinar|psych|paramedic|phlebotom|radiolog|clinical|"
                   r"nutrition|hygienist|care_aide|home_health|midwif|optometr|chiroprac|"
                   r"dietitian|emt"),
    ("education", r"teacher|professor|instructor|educat|tutor|childcare|preschool|"
                  r"librarian|teaching|faculty|principal_school|clergy"),
    ("security_safety", r"police|sheriff|security|guard|firefighter|safety|patrol|"
                        r"correctional|detective|protective"),
    ("transport_logistics", r"driver|truck|delivery|courier|pilot|transit|railroad|freight|"
                            r"logistic|bus_|taxi|chauffeur|material_mover|stock|warehouse|"
                            r"shipping|postal|mail_carrier"),
    ("service_food", r"cook|chef|food|server|waiter|waitress|bartender|barista|culinary|"
                     r"cosmetolog|hairdress|barber|hospitality|usher|dishwasher|"
                     r"flight_attendant|concierge|baker|butcher|meat_"),
    ("sales_retail", r"sales|retail|cashier|merchand|real_estate|teller|counter_worker|"
                     r"buyer|store|purchasing"),
    ("manual_production", r"construction|laborer|production|manufactur|machin|weld|"
                          r"electric|plumb|carpenter|mechanic|installer|repair|assembl|"
                          r"janitor|cleaner|maid|housekeep|packag|mover|fabricat|operator|"
                          r"maintenance|farm|agricultur|landscap|painter|roofer|mason|"
                          r"grounds|extraction|logging|miner|pest_control|hvac|"
                          r"inspector|tester|sorter|sampler|weigher|metal_work|plastic_work|"
                          r"grinder|sewing|tailor|upholster"),
    ("professional_tech", r"develop|engineer|software|computer|programm|analyst|scientist|"
                          r"architect|designer|research|lawyer|attorney|accountant|auditor|"
                          r"consultant|technician|writer|journalist|translat|data_|actuar|"
                          r"statistic|economist|editor|web_|information_security|paralegal|"
                          r"business_operations|financial_advisor|mathematic|surveyor|"
                          r"social_scientist|appraiser"),
    ("management", r"manager|executive|director|chief|officer|supervisor|administrator|"
                   r"ceo|president|management|founder|owner"),
    ("office_admin", r"secretary|administrative|clerk|office|receptionist|data_entry|"
                     r"bookkeep|customer_service|representative|payroll|human_resources|"
                     r"dispatcher|coordinator|assistant_admin|social_worker"),
]
_GENERAL = "general"

# 그룹별: core 도메인(독립 태스크 가중), work_ratio(체인이 사무성 업무일 확률).
# 도메인 키는 content_en.DOMAINS와 정확히 일치해야 함.
OFFICE_CATS = {"office_admin", "professional_tech", "management", "sales_retail", "education"}
CAT_CONFIG: dict[str, dict] = {
    "healthcare":          {"domains": ["Health/Medical", "Family/Care", "Errands/Shopping", "Learning/Self-dev"], "work_ratio": 0.0},
    "education":           {"domains": ["Learning/Self-dev", "Family/Care", "Work/Office", "Social"], "work_ratio": 0.40},
    "security_safety":     {"domains": ["Home/Repairs", "Health/Medical", "Errands/Shopping", "Fitness/Hobby"], "work_ratio": 0.0},
    "transport_logistics": {"domains": ["Errands/Shopping", "Home/Repairs", "Health/Medical", "Family/Care"], "work_ratio": 0.0},
    "service_food":        {"domains": ["Errands/Shopping", "Family/Care", "Home/Repairs", "Health/Medical"], "work_ratio": 0.0},
    "sales_retail":        {"domains": ["Clients/Contacts", "Work/Office", "Digital/Booking", "Finance/Bills"], "work_ratio": 0.50},
    "manual_production":   {"domains": ["Home/Repairs", "Errands/Shopping", "Health/Medical", "Family/Care"], "work_ratio": 0.0},
    "professional_tech":   {"domains": ["Work/Office", "Learning/Self-dev", "Clients/Contacts", "Digital/Booking"], "work_ratio": 0.55},
    "management":          {"domains": ["Work/Office", "Clients/Contacts", "Finance/Bills", "Social"], "work_ratio": 0.60},
    "office_admin":        {"domains": ["Work/Office", "Clients/Contacts", "Digital/Booking", "Learning/Self-dev"], "work_ratio": 0.55},
    "jobseeker_senior":    {"domains": ["Learning/Self-dev", "Admin/Paperwork", "Health/Medical", "Family/Care"], "work_ratio": 0.0},
    _GENERAL:              {"domains": ["Family/Care", "Health/Medical", "Home/Repairs", "Social"], "work_ratio": 0.0},
}

_COMPILED = [(c, re.compile(p)) for c, p in CATEGORIES]


def categorize(occ: str) -> str:
    occ = (occ or "").lower()
    for cat, pat in _COMPILED:
        if pat.search(occ):
            return cat
    return _GENERAL


CATEGORY_NAMES = [c for c, _ in CATEGORIES] + [_GENERAL]
