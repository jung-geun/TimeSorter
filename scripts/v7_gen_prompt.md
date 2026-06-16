You are generating Korean to-do scheduling training data. Work in repo /mnt/hdd/WD_8TB/code/TimeSorter.

STEP 1 — Read the skeleton slice file: `{SKEL_FILE}`
Each line is a JSON object with keys: id, today, today_display, persona_name, persona_label, gen_lines (array), facts (string).

STEP 2 — For EACH skeleton row, produce one output object {"id", "tasks", "chosen"}:

(A) "tasks": an array of Korean to-do strings, one per gen_line, SAME order and SAME count as gen_lines.
   - Every task must be a CONCRETE, realistic to-do specific to that row's persona (occupation/age in persona_label). No filler/placeholder words — never "별도", "배경", "할일 완료", "기타 업무", generic "업무". For 무직/elderly personas, use age-appropriate everyday tasks (병원 예약, 약 복용, 공과금 납부, 장보기, 가족 연락 등).
   - If a gen_line says "마감 YYYY-MM-DD HH:MM (텍스트에 일시 포함)", embed that date/time naturally (e.g. "(10/14 13:00까지)").
   - CHAIN STEPS (gen_line "체인 N의 K단계"): write each as one step of a multi-step workpiece so that a reader seeing ONLY the task texts can identify which tasks form the chain AND their order. Requirements:
       • All steps of one chain share the SAME concrete deliverable/subject.
       • Each step explicitly consumes the previous step's output ("작성한 ~", "검토한 ~", "확정된 ~").
       • THE FINAL STEP (the one carrying the deadline) MUST ALSO name the chain's deliverable + a completing action. e.g. "완성한 분기 보고서 임원진 발송 (10/14 14:00까지)", "검증 마친 부가세 신고서 세무서 전자 신고 (12/11 15:00까지)".
       • ABSOLUTELY FORBIDDEN as a final step: generic phrases with no deliverable — "후속 조치 사항 최종 확인 완료", "최종 확인 완료", "마무리 처리", "후속 조치". These make the chain end unrecoverable from text and are the #1 rejection cause.
       • NEVER write "체인", "1단계", "2단계" literally — express order through work content only.
   - Example 4-step chain: "분기 매출 보고서 초안 작성" → "작성한 초안 데이터 검증" → "검증된 보고서 팀장 피드백 반영" → "최종 보고서 임원진 발송 (10/14 14:00까지)".
   - If two chains exist (체인 1 and 체인 2), use clearly DIFFERENT deliverables/topics so they never blend.
   - "마감 없음": no date/time in text.
   - No duplicate subjects across unrelated tasks.
   - DIVERSITY (critical): author EACH row's tasks individually and freshly. Do NOT write a Python generator that fills tasks from a small set of reusable per-occupation "task banks" — that produces the SAME phrasings over and over across rows and ruins the dataset. Vary deliverables, verbs, and domains row to row.
   - NO DUPLICATE TASK within a single row: every task text in a row must be distinct (even ignoring the deadline parenthesis). Two chain steps must never share the same wording.

(B) "chosen": a JSON object {"tasks":[{"id":N,"text":"..."}],"priority_order":[...],"scores":[{"task_id":N,"urgency":1-5,"importance":1-5,"dependency":1-5,"time_constraint":1-5,"reason":"..."}]}
   Rules the chosen MUST satisfy (machine-verified by verify_chosen):
   - tasks: id 1..N matching the order of your generated texts; text = the same strings.
   - priority_order: a permutation of all ids.
   - CHAIN: all tasks of the same chain group CONTIGUOUS in priority_order and in step order (step1 ... final). Use the facts block to know each task's chain group + step.
   - Every NON-FINAL chain step must have dependency >= 4.
   - Same-day deadlines: earlier time ranked before later time.
   - A task with "마감 없음" must NOT be ranked #1.
   - scores: realistic 1-5 per axis; reason = short Korean justification (do NOT mention "체인/단계"; justify by work logic).
   The "facts" string in each skeleton row is your ground-truth grading reference — follow it exactly.

STEP 3 — Write all output objects to `{OUT_FILE}`, ONE JSON object per line (JSONL). "chosen" must be a nested JSON object (NOT a stringified string). Process every row in the slice.

After writing, report: number of rows written, and paste 2 full task-text arrays (different personas) for spot-check. Keep your final message short — it is data, not prose.
