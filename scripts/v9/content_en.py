"""v9 EN-US 콘텐츠 풀 — build_dataset_en.py가 사용하는 도메인/시나리오 텍스트(미국 현지화).

KR `build_dataset.py`의 콘텐츠 부분(DOMAINS/WORK_CHAINS/PERSONAL_CHAINS/RISK_CLAUSES/
MICRO_TOPICS/CHAIN_SUBJECTS)을 미국 생활/업무 맥락으로 치환. 로직은 build_dataset_en이 공유.
문화특화 치환: 김장→holiday meal prep, 연말정산→tax filing(1040/W-2), 인감증명→SSA/DMV,
정보처리기사→CompTIA, 종합건강검진→annual physical.

도메인 키는 occupations_en.CAT_CONFIG의 domains와 정확히 일치해야 함.
"""
from __future__ import annotations

import datetime as _dt

# 미국 동부 표준시(고정 오프셋 — KR의 KST처럼 단순화; DST는 향후 보강).
TZ = _dt.timezone(_dt.timedelta(hours=-5))
COUNTRY = "United States"

# 체인 domain_hint 상수 (build_dataset_en 로직이 참조)
WORK_DOMAIN = "Work/Office"   # 사무성 체인
MISC_DOMAIN = "Misc"          # 개인 체인/일반 fallback
RISK_DOMAINS = ("Work/Office", "Clients/Contacts")  # risk 부여 가능 도메인

DOMAINS = [
    "Health/Medical", "Finance/Bills", "Family/Care", "Learning/Self-dev", "Home/Repairs",
    "Social", "Admin/Paperwork", "Errands/Shopping", "Digital/Booking", "Fitness/Hobby",
    "Work/Office", "Clients/Contacts",
]
WORK_CHAINS = [
    "Project Deliverable", "Quarterly Performance Report", "Contract Review & Signing",
    "Event Planning & Execution", "Internal Audit Prep", "Training Material Creation",
    "Product Launch Prep", "Annual Budget Planning",
]
PERSONAL_CHAINS = [
    "Moving Prep", "Trip Planning", "Housewarming Prep", "Annual Physical Process",
    "Certification Exam Prep", "Government Paperwork", "Holiday Meal Prep", "Tax Filing Prep",
    "Wedding Planning", "Selling Used Items", "Pet Adoption Prep", "Kid's School Enrollment",
]
RISK_CLAUSES = [
    "late fee applies if not paid", "penalty for missing the deadline",
    "client escalation expected if delayed", "statutory filing deadline — no extension",
    "reservation auto-cancels if no response", "ineligible if not submitted on time",
]

# 도메인별 세부 소재 — 슬롯마다 순환 배정(같은 도메인도 다른 제목)
MICRO_TOPICS = {
    "Health/Medical": ["dental cleaning", "flu shot", "annual physical follow-up", "physical therapy session",
                       "eye exam", "blood pressure med refill", "dermatology visit", "therapy appointment",
                       "lab work appointment", "colonoscopy scheduling", "specialist referral", "vaccine booster"],
    "Finance/Bills": ["pay electric bill", "credit card payment", "renew auto insurance", "CD maturity rollover",
                      "HOA dues payment", "switch phone plan", "gather tax documents", "mortgage payment",
                      "check 401k contribution", "dispute a charge", "review dividend statement", "file insurance claim"],
    "Family/Care": ["take parent to doctor", "register kid for class", "dog grooming appointment", "plan family birthday dinner",
                    "visit grandparents", "kid's vaccine appointment", "anniversary gift for spouse", "coordinate family trip dates",
                    "daycare consultation", "organize parent's medications"],
    "Learning/Self-dev": ["GRE practice test", "online course module", "book club discussion prep", "certification practice exam",
                          "coding practice problems", "language conversation practice", "read seminar materials", "writing practice",
                          "summarize lecture notes", "study group assignment"],
    "Home/Repairs": ["schedule furnace inspection", "re-caulk the bathroom", "clean AC filters", "tidy the garage",
                     "replace light bulbs", "clean the washing machine", "assemble furniture", "patch wall cracks",
                     "fix window screen", "organize the fridge"],
    "Social": ["dinner with a friend", "attend club meetup", "coordinate reunion date", "attend a wedding",
               "neighbor housewarming visit", "host an online hangout", "coffee chat with mentor", "volunteer shift"],
    "Admin/Paperwork": ["renew driver's license at DMV", "request a birth certificate", "renew passport", "update address with USPS",
                        "vehicle registration renewal", "submit required forms", "file a permit application", "prepare visa documents"],
    "Errands/Shopping": ["weekly grocery run", "return a package", "pick up dry cleaning", "pick up prescription",
                         "gift wrapping drop-off", "mail a certified letter", "buy household essentials", "pick up flowers"],
    "Digital/Booking": ["restaurant reservation", "concert ticket purchase", "salon appointment", "book a flight",
                        "book a hotel", "telehealth appointment", "online class signup", "cancel a subscription"],
    "Fitness/Hobby": ["personal training session", "weekend hike plan", "swim lesson", "instrument practice",
                      "photography outing", "yoga class", "bike tune-up", "baking project"],
    "Work/Office": ["compile weekly status report", "prep for client meeting", "process expense receipts", "write up meeting minutes",
                    "clear out the email backlog", "review approval documents", "draft customer quote", "check inventory levels"],
    "Clients/Contacts": ["reply to customer inquiry", "coordinate vendor schedule", "follow up with new lead", "handle a complaint",
                         "renewal reminder call", "confirm delivery schedule", "request partner meeting", "process a support ticket"],
    "Misc": ["tidy up the house", "check pantry stock", "plan the weekly menu", "take out recycling"],
}

# 체인 테마별 구체 변형
CHAIN_SUBJECTS = {
    "Project Deliverable": ["new app spec doc", "data analysis report", "marketing campaign brief", "UX improvement proposal"],
    "Quarterly Performance Report": ["Q2 sales report", "team performance dashboard", "cost-savings results", "regional sales analysis"],
    "Contract Review & Signing": ["new supplier contract", "lease agreement", "outsourcing service contract", "annual maintenance contract"],
    "Event Planning & Execution": ["company workshop", "client seminar", "product launch event", "anniversary event"],
    "Internal Audit Prep": ["accounting audit response", "security review", "quality certification audit", "safety inspection prep"],
    "Training Material Creation": ["new-hire onboarding guide", "safety training deck", "product manual", "customer support playbook"],
    "Product Launch Prep": ["beta launch checklist", "package design sign-off", "launch press release", "pre-order landing page"],
    "Annual Budget Planning": ["next-year department budget", "project budget", "marketing budget allocation", "capital investment plan"],
    "Moving Prep": ["studio to 2BR move", "office relocation", "moving parents' home", "out-of-state relocation"],
    "Trip Planning": ["family beach trip", "Japan itinerary", "business trip plus sightseeing", "Europe backpacking"],
    "Housewarming Prep": ["new-home housewarming", "post-move housewarming", "friends' game night", "dinner for parents"],
    "Annual Physical Process": ["annual wellness visit", "comprehensive checkup", "screening with bloodwork", "cardiac screening"],
    "Certification Exam Prep": ["CompTIA A+ exam", "real estate license exam", "PMP exam", "ServSafe certification"],
    "Government Paperwork": ["closing paperwork on a house", "LLC registration docs", "scholarship application forms", "benefits application"],
    "Holiday Meal Prep": ["Thanksgiving dinner", "holiday meal prep", "batch meal cooking", "potluck dish prep"],
    "Tax Filing Prep": ["1040 filing prep", "gather W-2 and 1099s", "organize medical receipts", "compile charitable receipts"],
    "Wedding Planning": ["book the venue", "honeymoon planning", "design invitations", "registry setup"],
    "Selling Used Items": ["sell used appliances", "closet cleanout sale", "pre-move decluttering", "sell old books"],
    "Pet Adoption Prep": ["puppy adoption prep", "cat supplies setup", "register with a vet", "pet insurance signup"],
    "Kid's School Enrollment": ["elementary enrollment", "daycare admission", "middle school registration", "college application docs"],
}

# 영어 placeholder 제목 거부 패턴(assemble 게이트) — KR의 _GENERIC 영어판
GENERIC_TITLE = (r"^(task|work|to-?do|do\s|process|handle|action|misc|"
                 r"general\s+work|various\s+tasks?)(\s+\d+)?$")
