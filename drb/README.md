# drb

Notebooks for Deep Research Bench runs against research endpoints you host yourself, plus the grading steps. Each run writes its results to a JSONL file.

Bring your own endpoint and data: the run notebooks need your own research endpoint (`BENCHMARK_API_KEY` and the `<MODAL_WORKSPACE>` and `<MODAL_APP_NAME>` placeholders in their URLs). The grading notebooks read the answer files the run notebooks write in the working directory, `drb_quick_scan_results.jsonl` and `drb_v4_deep_results.jsonl` (JSONL, one `{"id", "prompt", "article"}` object per line). No data ships with this folder.

- `drb-research-pro.ipynb`: DRB Research Pro run. Research submit refuses today (`RESEARCH_COST_BOUNDED = False`).
- `drb-quick-scan.ipynb`: Quick scan. Research submit refuses today (`RESEARCH_COST_BOUNDED = False`). Sends a set of queries to the research endpoint and writes the answers to a JSONL file.
- `drb-quick-scan-grading.ipynb`: Grades the quick-scan answers with the Deep Research Bench grading scripts. Refuses today (`GRADER_COST_BOUNDED = False`).
- `drb-deep-investigation-part-2.ipynb`: DRB Deep Investigation, part 2. A one-query smoke test against the deep-investigation endpoint, then the query run. The submit cells refuse today (`RESEARCH_COST_BOUNDED = False`), and so does the grading step (`GRADER_COST_BOUNDED = False`).
- `drb-deep-investigation-grading.ipynb`: Grades the Deep Investigation answers with the Deep Research Bench grading scripts. Refuses today (`GRADER_COST_BOUNDED = False`).

## Spending limits

These notebooks spend money on your own accounts and keys. See "Spending limits" in the root `README.md` (`../README.md`) for the budget cap (`RUN_BUDGET_USD`, `JOB_WORST_CASE_USD`), the shared ledger and lock (`SPEND_LEDGER_DIR`), and which notebooks currently refuse to run until a price is confirmed.

None of the notebooks in this folder can spend today:

- The grading notebooks and the grading step of part 2 refuse (`GRADER_COST_BOUNDED = False`). The grader makes its own paid model calls in a subprocess that cannot be metered or bounded here, and a flat per-step reservation is not a worst-case bound. Before they install or run anything they also refuse if the grader checkout has modified tracked files or untracked executable or unexpected files. The allowed untracked data paths are listed in the root `README.md`. If grading is ever enabled, scores from the pinned GPT-era grader are not comparable to scores from the earlier Gemini-based grader.
- The Modal research-job submits refuse (`RESEARCH_COST_BOUNDED = False`) until an enforceable hosted per-job cost ceiling and cancellation are verified. Status polls stay available.
