#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
from collections import Counter

def analyze_dataset():
    parquet_path = "data/scheduler_ko.parquet"
    print(f"=== [1] {parquet_path} 로딩 ===")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"파일을 읽는 도중 오류가 발생했습니다: {e}")
        return

    print(f"데이터셋 크기: {df.shape}")
    print(f"컬럼 목록: {list(df.columns)}")
    print(f"결측치 통계:\n{df.isnull().sum()}\n")

    print("=== [2] 컬럼별 데이터 개요 ===")
    print(df.head(2))
    print("\n" + "="*50 + "\n")

    # 페르소나 정보 추출 및 통계 분석
    # df["persona"] 형식: "이름 (직업, 나이세)" -> "김지은 (대학생, 21세)"
    print("=== [3] 페르소나 다양성 및 메타데이터 통계 ===")
    
    ages = []
    occupations = []
    
    # scheduler_ko.parquet 에 저장된 persona 형태는: f"{persona_name} ({persona_row['occupation']}, {persona_row['age']}세)"
    # 예: "길동 (의사, 35세)"
    persona_pattern = re.compile(r"(.+)\s*\((.+),\s*(\d+)세\)")
    
    parsed_count = 0
    for idx, row in df.iterrows():
        p_str = row.get("persona", "")
        m = persona_pattern.search(p_str)
        if m:
            name, occ, age = m.groups()
            ages.append(int(age))
            occupations.append(occ.strip())
            parsed_count += 1
        else:
            # 매칭이 안 되는 경우 백업 파싱 시도
            # nemotron 데이터셋에서 원본 메타데이터가 있을 수 있으므로 체크
            pass

    print(f"성공적으로 파싱된 페르소나 수: {parsed_count}/{len(df)}")
    
    if ages:
        ages = np.array(ages)
        print(f"나이 분포:")
        print(f"  - 평균: {ages.mean():.1f}세")
        print(f"  - 최소: {ages.min()}세")
        print(f"  - 최대: {ages.max()}세")
        print(f"  - 연령대 분포:")
        age_groups = Counter((ages // 10) * 10)
        for g, count in sorted(age_groups.items()):
            print(f"    * {g}대: {count}명 ({count/len(ages)*100:.1f}%)")

    if occupations:
        print(f"\n가장 흔한 직업 Top 10:")
        occ_counts = Counter(occupations)
        for occ, count in occ_counts.most_common(10):
            print(f"  - {occ}: {count}명")

    print("\n=== [4] 한국어 포맷 검증 ===")
    # 1) [할일] - [이유] 패턴 검증
    # prompt 검증: [이름 씨의 오늘의 할 일 목록]\n- [할일 1] ...
    # chosen 검증: 1) [할일 1] - [이유]\n2) ...
    
    format_failures = 0
    pattern_chosen = re.compile(r"^\d+\)\s+.+?\s+-\s+.+$")
    
    for idx, row in df.iterrows():
        chosen = row["chosen"]
        lines = [line.strip() for line in chosen.split("\n") if line.strip()]
        
        valid_lines = 0
        for line in lines:
            if pattern_chosen.match(line):
                valid_lines += 1
                
        if len(lines) == 0 or valid_lines < len(lines):
            format_failures += 1
            if format_failures <= 3:
                print(f"[경고] 샘플 {idx} 형식 불일치 의심:")
                print(f"  원문:\n{chosen}")
                print(f"  전체 줄 수: {len(lines)}, 유효한 줄 수: {valid_lines}")
                print("-" * 30)

    print(f"포맷 미준수 의심 샘플 수: {format_failures}/{len(df)}")
    if format_failures == 0:
        print("-> [검증 결과] 모든 샘플이 SFT 정렬 포맷(숫자) + 하이픈) 형식을 완벽하게 준수하고 있습니다!")

    print("\n=== [5] 고품질 생성 샘플 자세히 보기 (3개 무작위 샘플) ===")
    np.random.seed(42)  # 재현성을 위한 시드
    sample_indices = np.random.choice(len(df), size=3, replace=False)
    
    for count, idx in enumerate(sample_indices, 1):
        row = df.iloc[idx]
        print(f"\n--- [샘플 {count}] (Original Index: {row.get('original_idx', idx)}) ---")
        print(f"페르소나: {row.get('persona', 'N/A')}")
        print(f"Source: {row.get('source', 'N/A')}")
        print("\n[번역 및 로컬라이징된 오늘의 할 일 목록 (Prompt)]")
        print(row["prompt"])
        print("\n[우선순위 정렬 및 페르소나 맞춤형 근거 (Chosen)]")
        print(row["chosen"])
        print("-" * 60)

if __name__ == "__main__":
    analyze_dataset()
