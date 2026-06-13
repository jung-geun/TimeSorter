#!/usr/bin/env python3
"""발표자료 Part 1 통계 — n=30 벤치마크(정량) + 6쿼리 qualitative 통과 요약.

raw_outputs.json + outputs/eval_qwen35_*_n30.json 를 합쳐 stats.md 생성.
정량 정확도는 n=30 벤치마크에서, 정성 side-by-side는 6쿼리에서 가져온다.
"""
from __future__ import annotations

import json
from pathlib import Path

PRES = Path("presentation/01_model_comparison")
SCENARIOS = ["dated_mixed", "intraday", "risk", "relative", "dependency_chain"]
SC_LABEL = {"dated_mixed": "날짜혼재", "intraday": "당일시각",
            "risk": "리스크", "relative": "상대날짜", "dependency_chain": "의존성체인"}

# n=30 벤치마크 (정량 지표 — 통계적으로 의미있는 정확도)
N30 = {
    "base":     ("Qwen3.5-4B-Base (no FT)", "outputs/eval_qwen35_base_no_rlhf_n30.json"),
    "instruct": ("Qwen3.5-4B (no FT)",      "outputs/eval_qwen35_instruct_base_n30.json"),
    "sft":      ("Qwen3.5-4B + SFT v4",     "outputs/eval_qwen35_sft_q35_n30.json"),
}
DPO_MANUAL = {
    "clean": 27, "total": 30,
    "per_scenario": {
        "dated_mixed": {"n": 9, "violated": 0}, "intraday": {"n": 6, "violated": 0},
        "risk": {"n": 5, "violated": 1}, "relative": {"n": 4, "violated": 0},
        "dependency_chain": {"n": 6, "violated": 2},
    },
}


def _per_sc(d):
    return {sc: (v["n"] - v.get("violated", 0)) / v["n"] * 100
            for sc, v in d["per_scenario"].items()}


def main():
    raw = json.loads((PRES / "raw_outputs.json").read_text())

    lines = ["# Part 1 — 모델별 통계 (정량 정확도 + 정성 통과 요약)\n"]

    # ── 정량: n=30 벤치마크 ──
    lines += [
        "## 1-3-A. 정량 정확도 (held-out n=30, 통계적 지표)\n",
        "> 정확도는 시나리오별 30개 held-out 평가셋 기준. (6쿼리 정성 분석과 별개)",
        "> **instruct 0%는 추론 실패가 아니라 스키마 미준수**(tasks를 [{id,text}] 대신 "
        "[\"문자열\"]로 출력 → 파싱 실패). 파인튜닝의 1차 효과는 정확한 JSON 스키마 준수다. "
        "상세는 analysis.md 참조.\n",
        "| 모델 | 전체 | " + " | ".join(SC_LABEL[s] for s in SCENARIOS) + " |",
        "|------|------|" + "|".join("------" for _ in SCENARIOS) + "|",
    ]
    rows = []
    for key in ["base", "instruct", "sft"]:
        label, fpath = N30[key]
        d = json.loads(Path(fpath).read_text())
        rows.append((label, d))
    rows.insert(0, rows.pop(1))  # instruct 먼저
    # 순서: instruct, base, sft, dpo
    ordered = [
        ("Qwen3.5-4B (no FT)", json.loads(Path(N30["instruct"][1]).read_text())),
        ("Qwen3.5-4B-Base (no FT)", json.loads(Path(N30["base"][1]).read_text())),
        ("Qwen3.5-4B + SFT v4", json.loads(Path(N30["sft"][1]).read_text())),
        ("Qwen3.5-4B + DPO v5", DPO_MANUAL),
    ]
    for label, d in ordered:
        overall = d["clean"] / d["total"] * 100
        sc = _per_sc(d)
        cells = " | ".join(f"{sc.get(s, 0):.0f}%" for s in SCENARIOS)
        lines.append(f"| {label} | **{overall:.1f}%** | {cells} |")

    # ── 정성: 6쿼리 통과 요약 ──
    lines += [
        "\n## 1-3-B. 정성 분석 6쿼리 통과 요약 (side-by-side용)\n",
        "> 시나리오별 대표 1–2쿼리. 실제 출력·오류해설은 `analysis.md` 참조.\n",
        "| 쿼리(시나리오) | base | instruct | SFT | DPO |",
        "|---|---|---|---|---|",
    ]
    n_q = len(raw["queries"])
    for i in range(n_q):
        sc = raw["queries"][i]["scenario"]
        cells = []
        for key in ["base", "instruct", "sft", "dpo"]:
            r = raw["models"][key]["results"][i]
            cells.append("✅" if r["passed"] else f"❌{len(r['violations'])}")
        lines.append(f"| {SC_LABEL[sc]}(#{raw['queries'][i]['idx']}) | "
                     + " | ".join(cells) + " |")

    # 6쿼리 합계
    lines.append("\n**6쿼리 통과 합계:**\n")
    for key, label in [("base", "Base"), ("instruct", "Instruct"),
                       ("sft", "SFT v4"), ("dpo", "DPO v5")]:
        n_pass = sum(1 for r in raw["models"][key]["results"] if r["passed"])
        lines.append(f"- {label}: {n_pass}/{n_q}")

    (PRES / "stats.md").write_text("\n".join(lines) + "\n")
    print(f"[saved] {PRES / 'stats.md'}")


if __name__ == "__main__":
    main()
