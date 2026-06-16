You are a strict data auditor for Korean to-do scheduling training data. Work in /mnt/hdd/WD_8TB/code/TimeSorter.

Read `{IN}`. Each line: {"id": "...", "tasks": {"1": "텍스트", "2": "...", ...}}.

You are given ONLY the task texts — no labels, no ground truth. For each row, decide PURELY from the task texts which tasks form a dependency chain (a sequence where each step consumes the previous step's output / they share one deliverable and have an inherent execution order) and what that order is.

Rules:
- A dependency chain = 2+ tasks done in a specific order because each builds on the prior one's product (e.g. "보고서 초안 작성" → "작성한 초안 검토" → "검토본 발송"). The FINAL step typically names the deliverable + a completing action (발송/제출/완성/신고). Standalone tasks (errands, meetings, point-in-time checks) are NOT part of any chain.
- Identify chains using the task NUMBERS (keys). Output each chain as a list of numbers in EXECUTION ORDER (first step first). Task numbers are shuffled — rely on the producer→consumer wording, not key order.
- There may be 1 chain, 2 separate chains, or rarely none. Two independent chains → two separate lists. If two chains use similar connective wording, disambiguate by their shared deliverable/topic.
- Be objective: only group tasks whose text genuinely implies producer→consumer ordering. If membership/order is NOT inferable from text alone, do not force it — leave ambiguous tasks out.

Write your result to `{OUT}` — one JSON object per input row, SAME order:
{"id": "<same id>", "chains": [[step1_num, step2_num, ...], [...]]}
- "chains" = list of chains; each chain = list of task numbers (integers) in execution order. No chain → "chains": [].

Process EVERY row in the input. Use the Write/Bash tools to write the file — do NOT paste row contents into your reply. Final reply: ONLY the rows-written count and the 0/1/2-chain breakdown. Nothing else.
