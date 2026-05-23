# v2 데이터셋 분석 보고서
## SFT 데이터
- 총 행 수: **10958**
- chosen parse_lenient 통과율: **100.0%**

### 소스별 분포
| 소스 | 행 수 | 비율 |
|------|-------|------|
| regen_v2 | 5390 | 49.2% |
| orca_ko_filtered | 1969 | 18.0% |
| nemotron_v2 | 1899 | 17.3% |
| xlam_ko_translated | 500 | 4.6% |
| refusal_v2/music | 40 | 0.4% |
| refusal_v2/career | 40 | 0.4% |
| refusal_v2/travel | 40 | 0.4% |
| refusal_v2/pet | 40 | 0.4% |
| refusal_v2/weather | 40 | 0.4% |
| refusal_v2/diy | 40 | 0.4% |
| refusal_v2/translation | 40 | 0.4% |
| refusal_v2/trivia | 40 | 0.4% |
| refusal_v2/movie | 40 | 0.4% |
| refusal_v2/math | 40 | 0.4% |
| refusal_v2/science | 40 | 0.4% |
| refusal_v2/finance | 40 | 0.4% |
| refusal_v2/creative_writing | 40 | 0.4% |
| refusal_v2/fashion | 40 | 0.4% |
| refusal_v2/recipe | 40 | 0.4% |
| refusal_v2/news | 40 | 0.4% |
| refusal_v2/sports | 40 | 0.4% |
| refusal_v2/random_question | 40 | 0.4% |
| refusal_v2/math_word | 40 | 0.4% |
| refusal_v2/relationship | 40 | 0.4% |
| refusal_v2/geography | 40 | 0.4% |
| refusal_v2/history | 40 | 0.4% |
| refusal_v2/joke | 40 | 0.4% |
| refusal_v2/philosophy | 40 | 0.4% |
| refusal_v2/greeting | 40 | 0.4% |
| refusal_v2/medical | 40 | 0.4% |
| refusal_v2/game | 40 | 0.4% |
| refusal_v2/legal | 40 | 0.4% |
| refusal_v2/small_talk | 40 | 0.4% |
| refusal_v2/coding | 40 | 0.4% |

### 페르소나 분포 (상위 15)
| 페르소나 | 행 수 |
|---------|------|
| 부모 | 932 |
| 프리랜서 | 932 |
| 직장인 | 912 |
| 학생 | 907 |
| 창업자 | 617 |
| 의료진 | 607 |
| 연구자 | 601 |
| 시니어 | 593 |
| 연구원 | 312 |
| 마케터 | 298 |
| 스타트업 창업자 | 296 |
| 디자이너 | 293 |
| 김순복 (무직, 68세) | 2 |
| 신정숙 (무직, 63세) | 2 |
| 임점연 (주방 보조원, 59세) | 2 |

### 4축 점수 통계
| 축 | 평균 | 표준편차 | 경고 |
|----|------|----------|------|
| urgency | 3.13 | 1.18 |  |
| importance | 3.48 | 1.2 |  |
| dependency | 2.39 | 1.38 |  |
| time_constraint | 3.46 | 1.49 |  |

## DPO 데이터
- 총 쌍 수: **15338**
- chosen parse_lenient 통과율: **100.0%**

### rejected 카테고리 분포
| 카테고리 | 쌍 수 | 비율 |
|---------|-------|------|
| bad_scores | 3692 | 24.1% |
| urgency_only | 3675 | 24.0% |
| shallow_reason | 3664 | 23.9% |
| invalid_json | 3587 | 23.4% |

### 페르소나 분포 (상위 10)
| 페르소나 | 쌍 수 |
|---------|------|
| 창업자 | 1027 |
| 프리랜서 | 1019 |
| 부모 | 996 |
| 연구자 | 992 |
| 학생 | 992 |
| 직장인 | 990 |
| 의료진 | 986 |
| 시니어 | 980 |
| 주정숙 (건물 경비원, 69세) | 4 |
| 장명옥 (간호조무사, 60세) | 4 |
