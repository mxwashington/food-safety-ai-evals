# Regulatory and food-safety benchmark notebooks

Notebooks for benchmarking language models on regulatory and food-safety tasks. They cover regulatory-intelligence benchmarks, the training-data pipeline for a food-safety question-and-answer model, Deep Research Bench runs, and a food-safety recall analysis notebook with its paper.

## Layout

- `benchmarks/`: 13 benchmark and writing bake-off notebooks
- `fsqa-pipeline/`: 9 notebooks for phases 1 to 5 of the food-safety question-and-answer pipeline
- `drb/`: 5 notebooks for Deep Research Bench runs and grading
- `fsri/`: a recall analysis notebook, its paper PDF, and the paper's README

Each folder has a README.md with one line per notebook.

## Licenses

- Code in the notebooks: MIT, see `LICENSE`.
- Prompts and results: CC-BY-4.0, see `LICENSE-DATA`.

Result outputs were cleared before publication, so the notebooks carry prompts and method, not result tables or charts.

## Opening the notebooks

Open any `.ipynb` file in Jupyter, JupyterLab, or a notebook viewer that reads nbformat 4.

Several notebooks call hosted research endpoints that you deploy yourself; their URLs carry placeholders such as `<MODAL_WORKSPACE>` and `<MODAL_APP_NAME>`. Several read API keys from environment variables. Set your own keys and endpoints before running anything that calls a model or an API. Some notebooks also read data files or helper modules that this repository does not ship; where a notebook needs them, a "Bring your own data" note at its top names the files, where they go and their format.

## Notebooks that spend money

Every notebook that makes a paid model call or starts a paid job routes it through a spend guard:

- **Run budget.** Each run is capped at 25.00 USD (`RUN_BUDGET_USD`). Each attempt is reserved at its worst case before it is sent and settled after, including retries and failed calls. Run ledgers are `spend_ledger.jsonl` files under `spend_ledger` at the repo root (see "Spending limits" below).
- **Refusal.** A call is refused before it is sent if its model has no listed price in the notebook, has no output cap, or would take the run past the budget. A refused call halts the run.
- **One runner.** One shared `runner.lock` in `spend_ledger` at the repo root. A second runner, from any notebook, refuses to start while the lock exists. A stale lock is never cleared automatically; delete it by hand after confirming no runner is alive.
- **Grading.** The DRB grading notebooks refuse to grade at all today (`GRADER_COST_BOUNDED = False`). Before they would install or run anything they also check the `deep_research_bench` checkout: it must be at the pinned full commit SHA `GRADER_REVISION`, with no modified tracked files and no untracked executable or unexpected files.
- **Modal submits.** The Modal research-job submits refuse today (`RESEARCH_COST_BOUNDED = False`). The flat `JOB_WORST_CASE_USD` reservation (default 2.00 USD) is still taken in the same ledger before a submit, but it is not a worst-case bound, so it is not treated as the cost control.

No price is guessed. A notebook that calls a model whose listed price is not confirmed in its price table refuses to run and names the model.

## Modal secret names

These are names only. No secret value is stored in this repository.

| Name | Used for |
|---|---|
| `your-modal-secret` | The Modal secret object the notebooks attach (a placeholder name). Its keys supply the variables below. |
| `OPENROUTER_API_KEY` | Bearer key for OpenRouter requests. |
| `BENCHMARK_API_KEY` | Sent in the body of requests to the Modal research endpoints and in status polls. |
| `JINA_API_KEY` | Read by the grader's scrape step in the DRB fact-grading run (`drb-quick-scan-grading`). |
| `AWS_REGION` | Region for the Bedrock client in recall-bench-v201 (default `us-east-1`). |
| `BEDROCK_RETENTION_CONFIRM` | Opt-in gate for the Bedrock data-retention change in recall-bench-v201. Must equal `account:region`. |

AWS credentials are read by boto3's default credential chain, and the code sets no variable name for them.

## Spending limits

These notebooks spend money on your own accounts and keys (OpenRouter, OpenAI, AWS Bedrock, Modal, and the key the grader's model provider needs). The cap protects you, the reader.

How the cap works:

- Every paid attempt is reserved in a ledger at its worst case before it is sent, and settled after. The run stops when a reservation would pass the cap.
- `RUN_BUDGET_USD` is the cap per run (default 25.00). `JOB_WORST_CASE_USD` is the amount reserved for each Modal research submit (default 2.00). It is an accounting figure, not a worst-case bound. Change either with an environment variable of the same name. A value that is set must be a finite number above 0, or the notebook refuses to start.
- Direct OpenAI and Bedrock calls reserve the listed output price times max_tokens plus an input estimate. OpenRouter calls reserve the same way and also carry `provider.max_price` and usage accounting.
- SDK retries are disabled (`max_retries=0` for the OpenAI client, `max_attempts` 0 for boto3), so every attempt is reserved. A retry made by a notebook's own loop is reserved again.
- The ledger and one shared `runner.lock` live in `spend_ledger` at the repo root. Set `SPEND_LEDGER_DIR` to put them elsewhere, for example on a Modal volume. When it is not set and there is no `.git` above the working directory, the notebook refuses to start.
- Only one paid run can hold the lock at a time. A stale lock is never cleared automatically: confirm no runner is alive, then delete it by hand.

Prices and pins:

- Notebooks that pay per token (OpenRouter, OpenAI, Bedrock) read prices from a price table near the top of the notebook. Each figure is the highest listed endpoint price on the date given there. The DRB submit notebooks have no price table: they would reserve a flat `JOB_WORST_CASE_USD` per job, which is not a worst-case bound, so they refuse (see "Notebooks that cannot spend at all"). A notebook with an unconfirmed price refuses to run and names the model.
- The three grading notebooks (`drb-quick-scan-grading`, `drb-deep-investigation-grading` and the grading step of `drb-deep-investigation-part-2`) refuse unless the `deep_research_bench` checkout is at the pinned commit `GRADER_REVISION`. That is the upstream commit at the time of the pin (2026-11-05). That upstream commit is from after 11 May 2026, when upstream switched its evaluator from Gemini-2.5 to GPT-5.5 (its README keeps the earlier Gemini-2.5 code on a `Gemini-2.5` branch), so grading with it is not comparable to Gemini-based scores. All three grading notebooks follow the pinned GPT-based code. They read `LLM_BACKEND` (`openrouter` by default, or `openai`), take the key from `OPENROUTER_API_KEY` or `OPENAI_API_KEY`, refuse if `OPENROUTER_BASE_URL` or `OPENAI_BASE_URL` points anywhere but the official host, and refuse unless every model the grader would call has a confirmed price. These checks come before the cost refusal described below, which stops grading regardless. If grading is ever enabled, scores from them are not comparable to scores from the earlier Gemini-based grader.
- Modal secret names appear as names only.

Limits that cannot be metered, and what the notebooks do about them:

- The external grader makes its own paid model calls inside a subprocess. They cannot be metered or bounded here, and a flat per-step reservation is not a worst-case bound. The grading notebooks refuse.
- OpenRouter web-search fees and search-result tokens are billed apart from tokens and are not in the token reservation. The search paths refuse. Requests without search are unchanged.
- A Modal research job keeps running and billing after it is accepted, and a polling timeout is not a cancellation. The research-job submits refuse.

Notebooks that cannot spend at all today, and what must be true before the constant may be set to `True`:

- `drb/drb-quick-scan-grading.ipynb`, `drb/drb-deep-investigation-grading.ipynb` and the grading step of `drb/drb-deep-investigation-part-2.ipynb` (`GRADER_COST_BOUNDED = False`): the cost of every paid model call the grader makes inside its subprocess, retries included, must be bounded and enforced before it is dispatched.
- Web-search paths of `fsqa-pipeline/phase1-eval-gen-v2.ipynb`, `fsqa-pipeline/phase1-eval-gen-v3.ipynb` and `fsqa-pipeline/phase5-testset-rebuild.ipynb` (`SEARCH_COST_BOUNDED = False`): search uses, search results and search-result context tokens must be bounded and priced into the reservation made before each request. Without search these notebooks still run as before; in v2 and v3 set `USE_WEB_SEARCH = False`.
- Modal research-job submits in `drb/drb-quick-scan.ipynb`, `drb/drb-research-pro.ipynb` and the submit cells of `drb/drb-deep-investigation-part-2.ipynb` (`RESEARCH_COST_BOUNDED = False`): an enforceable per-job cost ceiling and a working way to cancel a running job on the hosted side must be verified. Status polls are not paid and stay available.

Notebooks that run only after a check:

- `benchmarks/bakeoff-r6-nemotron.ipynb` runs only after its provider probe cell succeeds and the response names a provider on the allowlist. A missing, empty or non-string provider name counts as a failure. The probe result is tied to the model ids and the allowlist it was taken with, so an older result cannot be reused.

Grader checkout rules, checked before the grading notebooks install or run anything:

- No modified, deleted, staged or mode-changed tracked file, and no nested `.git` or symlink.
- Untracked files (ignored ones included) are allowed only as plain data files without an executable bit: `.jsonl` or `.json` under `data/test_data/raw_data/` and `data/test_data/cleaned_data/`, and `.jsonl`, `.json`, `.txt` or `.log` under `results/`.
- The check runs again before the grader subprocess starts. The grader is run with `PYTHONDONTWRITEBYTECODE=1` so it does not leave untracked bytecode in the checkout.

Notebooks that refuse to run until a price is confirmed, as of 2026-11-05:

- `benchmarks/bakeoff-r3-legacy.ipynb`: no price for `anthropic/claude-3-opus`.
- `benchmarks/recall-bench-v201.ipynb`: no price for `us.anthropic.claude-fable-5-1` on Bedrock.

## Grader revision

Each grading notebook names the grader's default model ids in `_GRADER_BACKENDS`. `RACE_MODEL` and `CLEAN_MODEL` (and `FACT_MODEL` in `drb-quick-scan-grading`) can be overridden by environment variable, but only to a model listed in that notebook's `PRICE`; any other model is refused. `GRADER_REVISION` in the grading notebooks is the commit those notebooks check from now on. The DRB run notebooks check out the same commit right after a fresh clone of `deep_research_bench`, and on every run they refuse to read the queries unless the checkout is at that commit; they never move an existing checkout.

## Running the tests

The tests exec the notebooks' real code cells with the network, subprocesses and paid clients stubbed, and check that no paid request goes out on a refusal path. They use the standard library only and make no network calls:

```
python3 -m unittest discover -s tests -v
```
