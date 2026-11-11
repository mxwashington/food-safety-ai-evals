"""Tests for the three DRB grading notebooks. Standard library only.

drb/drb-deep-investigation-grading.ipynb, drb/drb-quick-scan-grading.ipynb, drb/drb-deep-investigation-part-2.ipynb

A. Working-tree integrity: the real verify_grader_tree_clean and verify function of each notebook, run against
   a disposable git checkout built in a temporary directory (nothing is cloned, no network).
B. Fail closed (grading and research submits): with the shipped constants nothing paid goes out, no
   reservation is written, nothing is installed.
C. Positive: with the constant set True after its definition, the old path still reaches reserve_job and the stubbed
   subprocess or HTTP call exactly as before, so the guard is the only thing in the way.
"""
import json, os, re, subprocess, unittest

import nbharness
import drb_tree as T
from drb_tree import GRADING_NBS, GRADER_CELL, KEYS, FORBIDDEN_NAMES


def nb_source(key_or_path, idx):
    return T.cell_source(GRADING_NBS.get(key_or_path, key_or_path), idx)


def is_git(cmd):
    return isinstance(cmd, (list, tuple)) and os.path.basename(str(cmd[0])) == "git"


# ----------------------------------------------------------------------------------------------------------------------
# A. integrity
# ----------------------------------------------------------------------------------------------------------------------
class IntegrityMixin:
    KEY = None

    def setUp(self):
        self.parent = T.tmpdir()
        self.ledger = os.path.join(self.parent, "ledger")
        self.cell, self.verify, repo_const = GRADER_CELL[self.KEY]
        url = re.search(rf'^{repo_const} = "([^"]+)"', nb_source(self.KEY, self.cell), re.M).group(1)
        self.repo, self.sha = T.build_repo(self.parent, url)

    def check(self, fn_name="verify_grader_tree_clean", env=None):
        """Run the real function against the disposable checkout. Returns the exception or None."""
        holder = {}

        def call(ns, rec):
            try:
                ns[fn_name]()
                holder["exc"] = None
            except BaseException as e:
                holder["exc"] = e

        out = T.run(self.KEY, [self.cell, call], env=dict(env or {}), cwd=self.parent, allow_git=True, defs_only=True,
                    patch={"GRADER_REVISION": self.sha})
        self.rec = out.rec
        return holder["exc"]

    def assertClean(self, fn_name="verify_grader_tree_clean", env=None):
        exc = self.check(fn_name, env)
        self.assertIsNone(exc, f"expected a clean pass, got {exc!r}")
        self.assertEqual(self.rec.paid, [])

    def assertRefuses(self, fragment, fn_name="verify_grader_tree_clean"):
        exc = self.check(fn_name)
        self.assertIsInstance(exc, RuntimeError)
        msg = str(exc)
        self.assertIn("Refusing to grade", msg)
        self.assertIn(fragment, msg)
        self.assertIn("git -C deep_research_bench status", msg)       # prints the manual fix
        self.assertIn("never cleans or checks out", msg)
        for name in FORBIDDEN_NAMES:
            self.assertNotIn(name, msg)
        # nothing installed, no grader subprocess: only git reads happened
        self.assertEqual(self.rec.paid, [])
        self.assertTrue(all(is_git(c) for c in self.rec.subprocess), self.rec.subprocess)
        self.assertFalse(any("pip" in str(c) for c in self.rec.subprocess))
        self.assertFalse(any("deepresearch_bench_race" in str(c) for c in self.rec.subprocess))

    # --- clean
    def test_clean_tree_passes(self):
        self.assertClean()

    def test_clean_tree_passes_through_the_verify_function(self):
        self.assertClean(self.verify)
        self.assertTrue(any("status" in c for c in self.rec.subprocess if is_git(c)), "tree status was not read")

    def test_inherited_git_variables_do_not_redirect_the_check(self):
        self.assertClean(env={"GIT_DIR": "/nonexistent", "GIT_WORK_TREE": "/nonexistent", "GIT_INDEX_FILE": "/nonexistent"})

    # --- tracked changes
    def test_modified_tracked_file_refuses(self):
        T.write(self.repo, "utils/api.py", "KEY = 'x'\nimport os\n")
        self.assertRefuses("utils/api.py")

    def test_deleted_tracked_file_refuses(self):
        os.remove(os.path.join(self.repo, "deepresearch_bench_race.py"))
        self.assertRefuses("deepresearch_bench_race.py")

    def test_staged_change_refuses(self):
        T.write(self.repo, "utils/api.py", "KEY = 'staged'\n")
        T.git(self.repo, "add", "utils/api.py", home=self.parent)
        self.assertRefuses("utils/api.py")

    def test_staged_change_with_worktree_reverted_refuses(self):
        orig = open(os.path.join(self.repo, "utils/api.py")).read()
        T.write(self.repo, "utils/api.py", "KEY = 'staged'\n")
        T.git(self.repo, "add", "utils/api.py", home=self.parent)
        T.write(self.repo, "utils/api.py", orig)
        self.assertRefuses("utils/api.py")

    def test_mode_changed_file_refuses(self):
        os.chmod(os.path.join(self.repo, "requirements.txt"), 0o755)
        self.assertRefuses("requirements.txt")

    def test_assume_unchanged_edit_is_not_hidden(self):
        T.write(self.repo, "utils/api.py", "KEY = 'hidden'\n")
        T.git(self.repo, "update-index", "--assume-unchanged", "utils/api.py", home=self.parent)
        self.assertRefuses("utils/api.py")

    # --- untracked files
    def test_untracked_py_refuses(self):
        T.write(self.repo, "evil.py")
        self.assertRefuses("evil.py")

    def test_untracked_executable_in_data_dir_refuses(self):
        T.write(self.repo, "data/test_data/raw_data/model.jsonl", '{"id": 1}\n', mode=0o755)
        self.assertRefuses("data/test_data/raw_data/model.jsonl")

    def test_untracked_executable_in_results_refuses(self):
        T.write(self.repo, "results/race/m/race_result.txt", "1\n", mode=0o755)
        self.assertRefuses("results/race/m/race_result.txt")

    def test_ignored_py_refuses(self):
        T.write(self.repo, "local_hook.py")
        self.assertIn("local_hook.py", T.git(self.repo, "status", "--porcelain", "--ignored", home=self.parent))
        self.assertEqual(T.git(self.repo, "status", "--porcelain", home=self.parent).strip(), "")   # plain status hides it
        self.assertRefuses("local_hook.py")

    def test_other_code_like_untracked_files_refuse(self):
        for rel in ("run.sh", "x.pth", "x.so", "setup.cfg", "pyproject.toml", "conftest.py", "sitecustomize.py",
                    "Makefile", "utils/extra.py", "results/x/plugin.py", "data/test_data/raw_data/loader.py",
                    "data/test_data/raw_data/readme", "results/x/run.sh", "results/x/data.jsonl.py"):
            with self.subTest(rel=rel):
                self.setUp()
                T.write(self.repo, rel)
                self.assertRefuses(rel)

    def test_untracked_file_outside_allowed_dirs_refuses(self):
        for rel in ("notes.jsonl", "cleaned_data/model.jsonl", "data/other/model.jsonl", "data/test_data/model.jsonl",
                    "result/x.txt", "results.txt", "docs/a.json"):
            with self.subTest(rel=rel):
                self.setUp()
                T.write(self.repo, rel, '{"id": 1}\n')
                self.assertRefuses(rel)

    def test_wrong_extension_inside_allowed_dirs_refuses(self):
        for rel in ("data/test_data/raw_data/m.txt", "data/test_data/raw_data/m.log", "results/x/m.csv",
                    "results/x/M.JSONL"):
            with self.subTest(rel=rel):
                self.setUp()
                T.write(self.repo, rel)
                self.assertRefuses(rel)

    def test_nested_git_directory_refuses(self):
        os.makedirs(os.path.join(self.repo, "results/nested"))
        subprocess.run(["git", "init", "-q", os.path.join(self.repo, "results/nested")], check=True,
                       capture_output=True, env=T._git_env(self.parent))
        self.assertRefuses("results/nested")

    def test_nested_git_file_refuses(self):
        T.write(self.repo, "results/sub/.git", "gitdir: /elsewhere\n")
        self.assertRefuses("results/sub")

    def test_untracked_symlink_refuses(self):
        os.makedirs(os.path.join(self.repo, "results/x"))
        os.symlink("/etc/passwd", os.path.join(self.repo, "results/x/link.txt"))
        self.assertRefuses("results/x/link.txt")

    # --- allowed data files
    def test_untracked_jsonl_under_raw_data_is_allowed(self):
        T.write(self.repo, "data/test_data/raw_data/deep-investigation-v4.jsonl", '{"id": 1}\n')
        T.write(self.repo, "data/test_data/raw_data/sub/other.json", "{}\n")
        self.assertClean()

    def test_untracked_cleaned_data_json_is_allowed_but_not_code(self):
        T.write(self.repo, "data/test_data/cleaned_data/deep-investigation-v4.jsonl", '{"id": 1}\n')
        self.assertClean()
        T.write(self.repo, "data/test_data/cleaned_data/hook.py")
        self.assertRefuses("data/test_data/cleaned_data/hook.py")

    def test_untracked_results_files_are_allowed(self):
        T.write(self.repo, "results/race/deep-investigation-v4/race_result.txt", "score 1\n")
        for rel in ("results/a.jsonl", "results/a/b.json", "results/a/b.log"):
            T.write(self.repo, rel, "{}\n")
        self.assertClean()

    def test_passes_after_the_notebook_copy_and_grader_writes(self):
        # the order the notebook uses: check, install, copy data in, grader writes results, check again
        self.assertClean()
        T.write(self.repo, "data/test_data/raw_data/deep-investigation-v4.jsonl", '{"id": 1}\n', mode=0o644)
        T.write(self.repo, "results/race/deep-investigation-v4/race_result.txt", "score 1\n")
        T.write(self.repo, "results/fact/m/extracted.jsonl", "{}\n")
        self.assertClean(self.verify)

    def test_verify_function_covers_the_tree_check(self):
        T.write(self.repo, "evil.py")
        self.assertRefuses("evil.py", self.verify)

    def test_head_and_origin_are_still_checked(self):
        T.git(self.repo, "remote", "set-url", "origin", "https://example.invalid/other.git", home=self.parent)
        exc = self.check(self.verify)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("origin", str(exc))


class IntegrityDeep(IntegrityMixin, unittest.TestCase):
    KEY = "deep"


class IntegrityQuick(IntegrityMixin, unittest.TestCase):
    KEY = "quick"


class IntegrityPart2(IntegrityMixin, unittest.TestCase):
    KEY = "part2"


# ----------------------------------------------------------------------------------------------------------------------
# Whole-notebook flows
# ----------------------------------------------------------------------------------------------------------------------
class FlowBase(unittest.TestCase):
    KEY = None
    RESULTS = {"deep": "drb_v4_deep_results.jsonl", "quick": "drb_quick_scan_results.jsonl",
               "part2": "drb_v4_deep_results.jsonl"}

    def setUp(self):
        self.parent = T.tmpdir()
        self.ledger = os.path.join(self.parent, "ledger")
        self.env = dict(KEYS, SPEND_LEDGER_DIR=self.ledger)
        T.write(self.parent, self.RESULTS[self.KEY], '{"id": 1, "prompt": "p", "article": "a"}\n')
        self.verify = GRADER_CELL[self.KEY][1]

    @property
    def steps(self):                  # the code cells that make one grading entry, in order
        return {"deep": [0], "quick": [0], "part2": [0, 4]}[self.KEY]

    def run_flow(self, steps, patch=None, calls=None, **kw):
        patch = dict(patch or {})
        if calls is not None:
            def counted(ns_unused=None):
                calls.append(1)
            patch[self.verify] = lambda: counted()
        return T.run(self.KEY, steps, env=self.env, cwd=self.parent, patch=patch, **kw)


class FailClosedMixin:
    """B. Shipped constants, a valid environment, the checkout verification stubbed OK: the guard is the only thing in
    the way, and it stops everything before any install, copy, reservation or subprocess."""

    def assert_nothing_happened(self, out):
        self.assertEqual(out.rec.paid, [])
        self.assertEqual(out.rec.subprocess, [])                   # no pip, no git, no grader
        self.assertEqual(T.reservations(self.ledger), [])
        self.assertFalse(os.path.exists(os.path.join(self.parent, "deep_research_bench")))   # nothing copied in
        self.assertFalse(os.path.exists(os.path.join(self.parent, "grading_log.txt")))

    def assert_grading_refusal(self, exc):
        self.assertIsInstance(exc, RuntimeError)
        msg = str(exc)
        self.assertTrue(msg.startswith("Refusing to grade:"), msg)
        for word in ("subprocess", "RACE", "retries", "not an upper bound", "64000", "bounded"):
            self.assertIn(word, msg)
        for name in FORBIDDEN_NAMES:
            self.assertNotIn(name, msg)

    def test_shipped_constant_is_false(self):
        for key in GRADING_NBS:
            cell = GRADER_CELL[key][0]
            src = nb_source(key, cell)
            self.assertEqual(len(re.findall(r"^GRADER_COST_BOUNDED = False\b", src, re.M)), 1, key)
            self.assertNotIn("GRADER_COST_BOUNDED = True", src)
        self.assertEqual(len(re.findall(r"^RESEARCH_COST_BOUNDED = False\b", nb_source("part2", 0), re.M)), 1)

    def test_grading_entry_refuses_first(self):
        out = self.run_flow(self.steps, calls=[])
        self.assert_grading_refusal(out.excs[-1] if self.KEY != "part2" else out.excs[-1])
        if self.KEY == "part2":      # cell 0 is the research smoke submit; it refuses on its own guard
            self.assertTrue(str(out.excs[0]).startswith("Refusing to submit a research job"))
        self.assert_nothing_happened(out)
        self.assertIs(out.ns["GRADER_COST_BOUNDED"], False)

    def test_refusal_names_no_ledger_variables(self):
        out = self.run_flow(self.steps)
        for e in out.excs:
            for name in FORBIDDEN_NAMES:
                self.assertNotIn(name, str(e))


class FailClosedDeep(FailClosedMixin, FlowBase):
    KEY = "deep"


class FailClosedQuick(FailClosedMixin, FlowBase):
    KEY = "quick"

    def test_fact_cell_refuses_first_even_with_everything_defined(self):
        # run cell 0 with the guard open (everything defined, ledger ready), then close it and run the FACT cell
        calls = []
        patch = {"GRADER_COST_BOUNDED": True}

        def close(ns, rec):
            ns["GRADER_COST_BOUNDED"] = False
            rec.paid.clear()
            rec.subprocess.clear()
            self.reserved_before = len(T.reservations(self.ledger))

        T.write(self.parent, "deep_research_bench/data/test_data/raw_data/quick-scan-v3.jsonl", "{}\n")
        out = self.run_flow([0, close, 1], patch=patch, calls=calls)
        self.assertIsNone(out.excs[0])
        self.assert_grading_refusal(out.excs[1])
        self.assertEqual(out.rec.paid, [])
        self.assertEqual(out.rec.subprocess, [])
        self.assertEqual(len(T.reservations(self.ledger)), self.reserved_before)

    def test_fact_cell_refuses_when_run_alone_after_the_refusal(self):
        out = self.run_flow([0, 1], calls=[])
        self.assert_grading_refusal(out.excs[0])
        self.assert_grading_refusal(out.excs[1])
        self.assert_nothing_happened(out)


class FailClosedPart2(FailClosedMixin, FlowBase):
    KEY = "part2"

    def test_research_submit_cells_refuse_before_any_request(self):
        out = self.run_flow([0, 1, 3], calls=[])
        for e in out.excs:
            self.assertIsInstance(e, RuntimeError)
            msg = str(e)
            self.assertTrue(msg.startswith("Refusing to submit a research job:"), msg)
            for word in ("per-job cost ceiling", "cancellation", "polling timeout is not cancellation", "not an upper bound"):
                self.assertIn(word, msg)
            for name in FORBIDDEN_NAMES:
                self.assertNotIn(name, msg)
        self.assertEqual(out.rec.paid, [])                         # no requests.post, no requests.get
        self.assertEqual(out.rec.subprocess, [])                   # not even the git clone in the submit cells
        self.assertEqual(T.reservations(self.ledger), [])
        self.assertIs(out.ns["RESEARCH_COST_BOUNDED"], False)

    def test_status_polling_is_still_unpaid_and_not_guarded(self):
        src = nb_source("part2", 1)
        poll = src[src.index("def poll_job"):src.index("total_time")]
        self.assertNotIn("require_bounded_research", poll)
        self.assertNotIn("reserve_job", poll)

    def test_guard_comes_before_reserve_and_post_in_every_submit_cell(self):
        for idx in (0, 1, 3):
            src = nb_source("part2", idx)
            for m in re.finditer(r"reserve_job\(\"(smoke|submit)\"\)", src):
                self.assertIn("require_bounded_research()", src[:m.start()], idx)
            first_post = src.index("requests.post")
            self.assertLess(src.index("require_bounded_research()"), first_post)
        # and in the loop cells, again before the per-attempt loop that holds the reserve and the post
        for idx in (1, 3):
            src = nb_source("part2", idx)
            self.assertLess(src.rindex("require_bounded_research()", 0, src.index("for attempt")), src.index("for attempt"))
            self.assertGreater(src.count("require_bounded_research()"), 1)


class PositiveMixin:
    """C. Constant patched True right after its definition: the old path runs as before."""

    def grade_patch(self):
        return {"GRADER_COST_BOUNDED": True}

    def assert_grader_path(self, out, calls, n_status_checks):
        self.assertIsNone(out.excs[-1], repr(out.excs[-1]))
        argvs = [list(c) for c in out.rec.subprocess if isinstance(c, (list, tuple))]
        pip = [i for i, c in enumerate(argvs) if c[0] == "pip"]
        grader = [i for i, c in enumerate(argvs) if "deepresearch_bench_race.py" in c]
        self.assertEqual(len(pip), 1)
        self.assertEqual(argvs[pip[0]][:3], ["pip", "install", "-r"])
        self.assertEqual(len(grader), 1)
        self.assertLess(pip[0], grader[0])
        self.assertEqual(argvs[grader[0]][-2], "--output_dir")
        self.assertEqual(len(calls), n_status_checks)            # the checkout check ran at the entry and before the run
        self.assertEqual(out.ns["_grader_env"].get("PYTHONDONTWRITEBYTECODE"), "1")   # no .pyc written into the checkout
        res = T.reservations(self.ledger)
        self.assertEqual([r["purpose"] for r in res], ["grade"])
        settled = [l for l in T.ledger_lines(self.ledger) if l.get("state") == "settled"]
        self.assertEqual(len(settled), 1)
        self.assertTrue(any("subprocess:python deepresearch_bench_race.py" in p for p in out.rec.paid))


class PositiveDeep(PositiveMixin, FlowBase):
    KEY = "deep"

    def test_old_path_runs_when_bounded(self):
        calls = []
        out = self.run_flow([0], patch=self.grade_patch(), calls=calls)
        self.assert_grader_path(out, calls, 2)
        self.assertIs(out.ns["GRADER_COST_BOUNDED"], True)
        self.assertTrue(os.path.exists(os.path.join(self.parent, "deep_research_bench/data/test_data/raw_data/deep-investigation-v4.jsonl")))


class PositiveQuick(PositiveMixin, FlowBase):
    KEY = "quick"

    def setUp(self):
        super().setUp()
        os.makedirs(os.path.join(self.parent, "deep_research_bench/data/test_data/raw_data"))

    def test_old_path_runs_when_bounded(self):
        calls = []
        out = self.run_flow([0], patch=self.grade_patch(), calls=calls)
        self.assert_grader_path(out, calls, 2)

    def test_fact_steps_run_with_a_tree_check_before_each(self):
        calls = []
        marks = []
        out = self.run_flow([0, lambda ns, rec: marks.append(len(calls)), 1], patch=self.grade_patch(), calls=calls)
        self.assertIsNone(out.excs[0])
        self.assertIsNone(out.excs[1], repr(out.excs[1]))
        steps = [c for c in out.rec.subprocess if isinstance(c, (list, tuple)) and c[:2] == ["python", "-u"]]
        self.assertEqual([c[3] for c in steps], ["utils.extract", "utils.deduplicate", "utils.scrape", "utils.validate", "utils.stat"])
        self.assertEqual(len(calls) - marks[0], 1 + 5)           # once at the entry, once before each of the 5 subprocesses
        res = T.reservations(self.ledger)
        self.assertEqual([r["purpose"] for r in res], ["grade"] * 6)


DRB_PIN = "f2735b5c3636759c22f9e936c0232de8bf545b67"   # GRADER_REVISION, and _DRB_PIN in the part 2 submit cells


class PositivePart2(PositiveMixin, FlowBase):
    KEY = "part2"

    def test_old_path_runs_when_bounded(self):
        calls = []
        out = self.run_flow([0, 4], patch=self.grade_patch(), calls=calls)
        self.assertTrue(str(out.excs[0]).startswith("Refusing to submit a research job"))
        argvs = [list(c) for c in out.rec.subprocess if isinstance(c, (list, tuple))]
        self.assertEqual(len([c for c in argvs if c[0] == "pip"]), 1)
        self.assertEqual(len([c for c in argvs if "deepresearch_bench_race.py" in c]), 1)
        self.assertIsNone(out.excs[1], repr(out.excs[1]))
        self.assertEqual(len(calls), 2)
        self.assertEqual([r["purpose"] for r in T.reservations(self.ledger)], ["grade"])

    def research_env(self):
        T.write(self.parent, "deep_research_bench/data/prompt_data/query.jsonl", '{"id": "q1", "prompt": "what is a recall"}\n')

    def test_smoke_submit_runs_when_research_is_bounded(self):
        out = self.run_flow([0], patch={"RESEARCH_COST_BOUNDED": True})
        self.assertIsNone(out.excs[0], repr(out.excs[0]))
        self.assertEqual([r["purpose"] for r in T.reservations(self.ledger)], ["smoke"])
        self.assertEqual([p for p in out.rec.paid if p.startswith("requests.")], ["requests.post", "requests.get"])

    def test_submit_loops_run_when_research_is_bounded(self):
        for cell in (1, 3):
            with self.subTest(cell=cell):
                self.setUp()
                self.research_env()
                # The submit cells check the deep_research_bench HEAD before reading the queries; the stubbed
                # rev-parse reports the pinned commit (_DRB_PIN in the notebook).
                out = self.run_flow([0, cell], patch={"RESEARCH_COST_BOUNDED": True},
                                    setup=lambda ns, rec: setattr(rec, "git_head", DRB_PIN))
                self.assertIsNone(out.excs[0], repr(out.excs[0]))
                self.assertIsNone(out.excs[1], repr(out.excs[1]))
                purposes = [r["purpose"] for r in T.reservations(self.ledger)]
                self.assertEqual(purposes, ["smoke"] + ["submit"] * 3)       # 3 attempts for the one query, each reserved
                self.assertEqual(out.rec.paid.count("requests.post"), 4)    # every post followed a reservation
                self.assertFalse(any(p.startswith("subprocess:") for p in out.rec.paid))


# ----------------------------------------------------------------------------------------------------------------------
# Whole-notebook flows over a real disposable checkout
# ----------------------------------------------------------------------------------------------------------------------
class RealCheckoutMixin:
    COPIED = "deep-investigation-v4.jsonl"
    """The real verify function and tree check run over a disposable checkout through the whole grading cell."""

    def setUp(self):
        super().setUp()
        cell, self.verify, repo_const = GRADER_CELL[self.KEY]
        url = re.search(rf'^{repo_const} = "([^"]+)"', nb_source(self.KEY, cell), re.M).group(1)
        self.repo, self.sha = T.build_repo(self.parent, url)

    def flow(self, extra_patch=None, steps=None):
        patch = {"GRADER_COST_BOUNDED": True, "GRADER_REVISION": self.sha}
        patch.update(extra_patch or {})
        return T.run(self.KEY, steps or self.steps, env=self.env, cwd=self.parent, patch=patch, allow_git=True)

    def test_clean_checkout_runs_through_and_passes_the_second_check(self):
        out = self.flow()
        self.assertIsNone(out.excs[-1], repr(out.excs[-1]))
        git_status = [c for c in out.rec.subprocess if is_git(c) and "status" in c]
        self.assertEqual(len(git_status), 2)                     # once at the entry, once again before the grader run
        self.assertTrue(os.path.exists(os.path.join(self.repo, "data/test_data/raw_data/" + self.COPIED)))
        self.assertEqual([r["purpose"] for r in T.reservations(self.ledger)], ["grade"])

    def test_file_planted_during_install_refuses_before_the_grader_runs(self):
        repo = self.repo

        class Planting:                  # replaces shutil in the notebook namespace: copies, then a stray file appears
            @staticmethod
            def copy(src, dst):
                import shutil as real
                real.copy(src, dst)
                T.write(repo, "sitecustomize.py", "import os\n")

        out = self.flow({"shutil": Planting})
        exc = out.excs[-1]
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("sitecustomize.py", str(exc))
        self.assertFalse(any("deepresearch_bench_race.py" in str(c) for c in out.rec.subprocess if not is_git(c)))
        self.assertEqual(T.reservations(self.ledger), [])

    def test_dirty_checkout_refuses_before_install(self):
        T.write(self.repo, "utils/api.py", "KEY = 'changed'\n")
        out = self.flow()
        exc = out.excs[-1]
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("utils/api.py", str(exc))
        self.assertFalse(any(not is_git(c) for c in out.rec.subprocess), out.rec.subprocess)   # no pip
        self.assertFalse(os.path.exists(os.path.join(self.repo, "data/test_data/raw_data/" + self.COPIED)))   # no copy
        self.assertEqual(out.rec.paid, [])
        self.assertEqual(T.reservations(self.ledger), [])


class RealCheckoutDeep(RealCheckoutMixin, FlowBase):
    KEY = "deep"


class RealCheckoutQuick(RealCheckoutMixin, FlowBase):
    KEY = "quick"
    COPIED = "quick-scan-v3.jsonl"

    def test_fact_cell_checks_the_tree_before_each_step(self):
        out = self.flow(steps=[0, 1])
        self.assertIsNone(out.excs[0], repr(out.excs[0]))
        self.assertIsNone(out.excs[1], repr(out.excs[1]))
        git_status = [c for c in out.rec.subprocess if is_git(c) and "status" in c]
        self.assertEqual(len(git_status), 2 + 1 + 5)


class RealCheckoutPart2(RealCheckoutMixin, FlowBase):
    KEY = "part2"



if __name__ == "__main__":
    unittest.main()
