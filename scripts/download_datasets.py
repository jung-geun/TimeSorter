#!/usr/bin/env python
"""외부 HuggingFace 데이터셋들을 로컬 data/ 디렉토리에 Parquet 형식으로 다운로드합니다.
"""
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

DATASETS_TO_DOWNLOAD = {
    "maywell/ko_Ultrafeedback_binarized": "data/ko_Ultrafeedback_binarized.parquet",
    "SJ-Donald/orca-dpo-pairs-ko": "data/orca-dpo-pairs-ko.parquet",
    "anakin87/events-scheduling": "data/events-scheduling.parquet",
    "nvidia/Nemotron-Personas-Korea": "data/nemotron_personas_korea.parquet",
}

def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, local_path in DATASETS_TO_DOWNLOAD.items():
        print(f"\n[다운로드 시작] {repo_id} -> {local_path} ...")
        try:
            # datasets 라이브러리를 통해 HF에서 데이터셋 로드
            ds = load_dataset(repo_id, split="train")
            
            # 로컬 Parquet 파일로 저장
            ds.to_parquet(local_path)
            print(f"[다운로드 완료] {repo_id} 저장 완료 ({len(ds)}개 샘플)")
        except Exception as e:
            print(f"[에러 발생] {repo_id} 다운로드 실패: {e}")

if __name__ == "__main__":
    main()
