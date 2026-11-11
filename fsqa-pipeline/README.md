# fsqa-pipeline

Notebooks for the food-safety question-and-answer training pipeline, in phase order. They cover question generation, generator selection, screening and confirmation, training-pair generation, and the test-set rebuild.

- `phase1-eval-gen.ipynb`: Phase 1, evaluation question generation. Generates 1,000 open-ended food-safety questions with reference answers from a teacher model.
- `phase1-eval-gen-v2.ipynb`: Phase 1, v2 (grounded). Each question is generated from the live text of a CFR section pulled from the eCFR API.
- `phase1-eval-gen-v3.ipynb`: Phase 1, v3. The grounded generation with a citation-normalization fix and retries on eCFR server errors.
- `phase2-model-eval.ipynb`: Phase 2. Scores a family of models at each effort level they accept on the Phase 1 question set, to pick the generator for training-pair synthesis.
- `phase2b-screen.ipynb`: Phase 2b. Screens the candidate configurations on one split of the Phase 1 question set to pick a shortlist for Phase 2c.
- `phase2c-confirm.ipynb`: Phase 2c. Confirms the shortlist winner as the teacher for Phase 4, with a judge that reads the actual CFR section.
- `phase3-generation.ipynb`: Phase 3. Tests generation rather than answering: inventing a realistic scenario grounded in a CFR section and writing a correct answer.
- `phase4-training-pairs.ipynb`: Phase 4. Generates the training corpus for the fine-tune from CFR-grounded scenarios and answers.
- `phase5-testset-rebuild.ipynb`: Phase 5. Rebuilds the Phase 1 test set in two jobs. The first QA-filters the existing questions, which were never filtered before.

## Bring your own data

No data ships with this repository. Each phase reads the output of the phases before it from the notebook volume under `./local-data` (for example `phase1_fsqa_eval/<run id>/questions.jsonl`). The top cell of each notebook from Phase 2 on names the exact files and folders it expects. Run the phases in order, or supply your own files in the same format.

## Spending limits

These notebooks spend money on your own accounts and keys. See "Spending limits" in the root `README.md` (`../README.md`) for the budget cap (`RUN_BUDGET_USD`), the shared ledger and lock (`SPEND_LEDGER_DIR`), and which notebooks currently refuse to run until a price is confirmed.

The web-search paths of `phase1-eval-gen-v2`, `phase1-eval-gen-v3` and `phase5-testset-rebuild` refuse (`SEARCH_COST_BOUNDED = False`). Search is billed apart from tokens, so the token reservation does not bound it. Requests without search are unchanged. In v2 and v3 set `USE_WEB_SEARCH = False` to run without search.
