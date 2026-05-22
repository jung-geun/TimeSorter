#!/usr/bin/env python
"""NVIDIA Nemotron-Personas-Korea 데이터셋을 연동한 
초정밀 한국어 스케줄러 SFT 데이터셋 생성 비동기 스크립트입니다.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# 비동기 OpenAI 클라이언트 초기화 (OPENAI_API_KEY 로드)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

aclient = AsyncOpenAI(api_key=api_key)

SYSTEM_PROMPT = """당신은 한국의 실제 인구 통계를 반영한 페르소나 맞춤형 스케줄 정렬 비서입니다.
지정된 페르소나의 배경, 연령, 직업, 말투(사투리 등)를 깊이 있게 반영하여 작업을 완수하십시오."""

USER_PROMPT_TEMPLATE = """[페르소나 프로필]
- 이름: {persona_name} ({age}세, {sex})
- 직업: {occupation}
- 거주지 및 문화적 배경: {district}, {cultural_background}
- 문체 스타일 및 성격: {professional_persona}

[영문 일정 목록 및 우선순위 정보]
영문 할 일 목록:
{english_events}

영문 원본 정답 우선순위:
{english_priority}

[임무]
1. 위의 '영문 할 일 목록'을 제공된 페르소나의 일상에 걸맞게 아주 자연스러운 한국어 문장의 '오늘의 할 일 목록'으로 로컬라이징(의역)하십시오. 
   - 예: 'Quantum computing guest talk' -> '양자 컴퓨터 영재 교육 설명회 참여' (부모 페르소나인 경우) 또는 '양자 컴퓨터 석학 강연 청강' (학생/연구원인 경우).
   - 페르소나의 어투(예: 연령대에 맞는 종결어미, 사투리 톤 등)를 '오늘의 할 일 목록' 제목이나 텍스트에 적극 반영하여 몰입감을 높이십시오.
2. 4축(긴급도·중요도·의존성·시간 제약)을 기준으로 해당 할 일들의 우선순위를 정렬하십시오. 
   - 각 정렬된 항목 뒤에 페르소나의 관점과 말투로 정렬한 타당한 이유를 한 줄씩 명시하십시오. (예: "1) ... - ...")
3. 반드시 아래의 JSON 형식만 엄격히 준수하여 출력하십시오. 다른 불필요한 설명이나 텍스트는 절대 덧붙이지 마십시오.

[출력 JSON 형식]
{{
  "prompt": "[{persona_name} 씨의 오늘의 할 일 목록]\\n- [할일 1]\\n- [할일 2]...",
  "chosen": "1) [할일 1] - [이유]\\n2) [할일 2] - [이유]..."
}}
"""

async def translate_scenario_async(
    idx: int,
    english_prompt: str,
    english_priority: str,
    persona_row: dict,
    model: str = "gpt-4o-mini",
    max_retries: int = 3
) -> dict | None:
    persona_name = persona_row["persona"].split(" 씨는")[0].strip()
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        persona_name=persona_name,
        age=persona_row["age"],
        sex=persona_row["sex"],
        occupation=persona_row["occupation"],
        district=persona_row["district"],
        cultural_background=persona_row["cultural_background"],
        professional_persona=persona_row["professional_persona"],
        english_events=english_prompt,
        english_priority=english_priority
    )

    for attempt in range(max_retries):
        try:
            print(f"  [시작] 시나리오 {idx+1} ({persona_name} 씨, {persona_row['age']}세) 생성 중... (시도 {attempt+1}/{max_retries})")
            
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_completion_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            res_json = json.loads(content)
            
            # 자가 검증 (Evaluation Step)
            if "prompt" not in res_json or "chosen" not in res_json:
                raise ValueError("JSON에 필수 필드('prompt', 'chosen')가 누락되었습니다.")
                
            prompt_lines = res_json["prompt"].split("\n")
            chosen_lines = res_json["chosen"].split("\n")
            
            if len(prompt_lines) < 2 or len(chosen_lines) < 2:
                raise ValueError("생성된 프롬프트나 정렬 결과의 행 수가 너무 적습니다.")
            
            print(f"  [성공] 시나리오 {idx+1} 검증 통과!")
            return {
                "prompt": res_json["prompt"],
                "chosen": res_json["chosen"],
                "persona": f"{persona_name} ({persona_row['occupation']}, {persona_row['age']}세)",
                "source": "events-scheduling-nemotron",
                "original_idx": idx
            }
            
        except Exception as e:
            print(f"  [경고] 시나리오 {idx+1} 처리 에러: {e}")
            await asyncio.sleep(1.5 ** attempt)
            
    print(f"  [실패] 시나리오 {idx+1} 번역 실패 (재시도 횟수 초과)")
    return None

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 시나리오 수 제한")
    parser.add_argument("--out", default="data/scheduler_ko.parquet")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI 모델명")
    parser.add_argument("--concurrency", type=int, default=15, help="동시 실행 데스크 수")
    parser.add_argument("--random-state", type=int, default=42, help="페르소나 샘플링 랜덤 시드")
    args = parser.parse_args()

    # 1. 데이터 로드
    print("[1] 영문 시드 및 Nemotron 페르소나 데이터셋 로딩...")
    df_seed = pd.read_parquet("data/events-scheduling.parquet")
    df_nemotron = pd.read_parquet("data/nemotron_personas_korea.parquet")

    if args.limit:
        df_seed = df_seed.head(args.limit)

    num_samples = len(df_seed)
    print(f"  - 번역 대상 시드 시나리오: {num_samples}개")

    # 2. 페르소나 고유 샘플링 (--random-state로 다양한 페르소나 조합 생성 가능)
    print(f"[2] Nemotron 데이터셋에서 고유 페르소나 {num_samples}개 샘플링 (seed={args.random_state})...")
    df_personas_sampled = df_nemotron.sample(n=num_samples, random_state=args.random_state).reset_index(drop=True)
    
    # 3. 비동기 작업 큐 생성 및 배치 처리
    print(f"[3] 비동기 번역 & 가공 엔진 기동 (동시 처리 한도: {args.concurrency})...")
    sem = asyncio.Semaphore(args.concurrency)
    
    async def worker(idx, seed_row, persona_row):
        async with sem:
            english_prompt = seed_row["prompt"]
            # priority_events가 리스트인 경우 스트링으로 변환
            priority_list = seed_row["priority_events"]
            english_priority = ", ".join(priority_list) if isinstance(priority_list, list) else str(priority_list)
            
            return await translate_scenario_async(
                idx=idx,
                english_prompt=english_prompt,
                english_priority=english_priority,
                persona_row=persona_row.to_dict(),
                model=args.model
            )

    tasks = [
        worker(i, df_seed.iloc[i], df_personas_sampled.iloc[i])
        for i in range(num_samples)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 4. 성공한 결과만 정제 및 저장
    valid_results = [r for r in results if r is not None]
    print(f"\n[4] 저장 및 마무리... (성공률: {len(valid_results)}/{num_samples})")
    
    if valid_results:
        df_out = pd.DataFrame(valid_results)
        
        # 기존 parquet에 덮어씌움
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path, index=False)
        print(f"  - 최종 parquet 저장 완료: {out_path} ({len(df_out)}개 샘플)")
        
        # SFT 체크포인트가 있으면 삭제하여 꼬임 방지
        ckpt_path = out_path.with_suffix(".ckpt.jsonl")
        if ckpt_path.exists():
            ckpt_path.unlink()
            print("  - 기존 scheduler_ko.ckpt.jsonl 체크포인트 삭제 완료")
    else:
        print("  - [에러] 성공한 시나리오가 없습니다.")

if __name__ == "__main__":
    asyncio.run(main_async())
