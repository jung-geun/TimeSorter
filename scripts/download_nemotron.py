#!/usr/bin/env python
"""NVIDIA Nemotron-Personas-Korea 데이터셋을 로컬 Parquet 파일로 저장합니다.
"""
from pathlib import Path
from datasets import load_dataset

def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nemotron_personas_korea.parquet"
    
    print(f"[다운로드 시작] nvidia/Nemotron-Personas-Korea -> {out_path} ...")
    try:
        # 데이터셋 로드 (이미 캐싱되어 있음)
        ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train")
        
        # 로컬 Parquet 파일로 효율적으로 저장
        ds.to_parquet(str(out_path))
        print(f"[다운로드 완료] nvidia/Nemotron-Personas-Korea 저장 완료 ({len(ds)}개 샘플)")
    except Exception as e:
        print(f"[에러 발생] 다운로드 및 저장 실패: {e}")

if __name__ == "__main__":
    main()
