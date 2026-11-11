# benchmarks

Benchmark and writing bake-off notebooks. The benchmarks pair question sets with answer keys or rubrics and a run harness. The bake-off notebooks compare writing setups round by round, from one round to the next.

- `recall-bench-v2.ipynb`: Recall Bench v2, wide field. 25 frozen questions and an answer key, answered through OpenRouter and graded by two judge models. Bring your own data: the note at the top of the notebook lists the three files it expects.
- `recall-bench-v201.ipynb`: Recall Bench v2, wide field. Same opening cell and question set as `recall-bench-v2.ipynb`.
- `recall-bench-v202.ipynb`: Recall Bench v2, wide field. Same opening cell and question set as `recall-bench-v2.ipynb`.
- `recall-bench-2-0-3.ipynb`: Recall Bench 2.0.3. Same questions, key and judges as v2, with the output cap raised from 4,000 to 12,000 tokens so reasoning does not crowd out the answer.
- `allergen-bench-v1.ipynb`: Allergen Label-Consistency Bench v1. Given only the finished label, as text or as a photo, can a model tell whether it is inconsistent with the required allergen declaration?
- `allergen-bench-v101.ipynb`: Allergen Label-Consistency Bench v1, second file. Same question and three outcomes as `allergen-bench-v1.ipynb`.
- `complicity-bench-v1.ipynb`: Complicity Bench v1, legacy versus current models. When asked to write up a food-safety decision that should not have been made, does the model refuse or write the justification?
- `gpt-eval-9426.ipynb`: Evaluation run on the recall question set. The first code cell stages the data into a mounted volume, then discovers models and runs a smoke test. Bring your own data: the first cell lists the four files it expects.
- `writing-bakeoff-modal.ipynb`: Writing bake-off, round 2. Seven configurations and two briefs, 28 calls, with a blind grading step.
- `bakeoff-r3-legacy.ipynb`: Writing bake-off, round 3. Whether older models still write better than current ones. Four configurations and two briefs, 16 calls, without blind or automated grading.
- `bakeoff-r4-effort.ipynb`: Writing bake-off, round 4. An effort sweep: four models at three effort levels, one variable changed at a time.
- `bakeoff-r5-topics.ipynb`: Writing bake-off, round 5. A topic sweep: three configurations across eight short-form and two long-form topics, 30 calls.
- `bakeoff-r6-nemotron.ipynb`: Writing bake-off, round 6. A head-to-head of a newer open-weights model against a config carried over from round 5 on ten topics, with a data-path guard for endpoints that log prompts.

## Spending limits

These notebooks spend money on your own accounts and keys. See "Spending limits" in the root `README.md` for the budget cap (`RUN_BUDGET_USD`), the shared ledger and lock (`SPEND_LEDGER_DIR`), and which notebooks currently refuse to run until a price is confirmed.

- `bakeoff-r6-nemotron` runs only after its provider probe succeeded and returned an allowlisted provider. Missing provider metadata counts as a failure.
- `bakeoff-r3-legacy` and `recall-bench-v201` still refuse until prices are supplied.
