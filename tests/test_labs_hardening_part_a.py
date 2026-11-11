"""Offline checks for labs hardening part A. Standard library only. No network, no paid calls.

1. Every notebook that builds a path from RUN_ID validates RUN_ID first, with the _checked_run_id helper copied verbatim
   from complicity-bench-v1, or with one of the four stricter inline regex checks that were kept (each now after the
   None-only default and a non-empty-string check). Checked on the parsed source (ast), then by running the setup cells
   with a RUN_ID that tries to leave the ledger root and with "", 0 and False.
2. deep_research_bench is cloned only when missing, a fresh clone is checked out at the GRADER_REVISION that the
   grading notebooks verify, and every run refuses to read the DRB queries unless HEAD is at that commit.
"""
import ast, json, os, pathlib, re, sys, tempfile, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nbharness  # noqa: E402
from public_copy_exclusions import absent, present, skip_if_absent  # noqa: E402

REPO = nbharness.REPO
COMPLICITY = "benchmarks/complicity-bench-v1.ipynb"
NOTEBOOKS = sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*.ipynb"))

# Already guarded with _checked_run_id before part A.
STANDARD_BEFORE = ["benchmarks/complicity-bench-v1.ipynb", "benchmarks/gpt-eval-9426.ipynb"]
# Inline regex checks kept in place. Each runs after the None-only default and a non-empty-string check, so a blank or
# non-string RUN_ID ("", 0, False included) is refused there with RuntimeError before the regex sees it. Each regex
# accepts only names made of letters, digits, '_', '.' and '-', with no '..' name, so it refuses every string
# _checked_run_id refuses and more ('a/b', for one). Swapping in the helper would loosen them.
REGEX_LEFT = ["benchmarks/bakeoff-r5-topics.ipynb", "benchmarks/writing-bakeoff-modal.ipynb",
              "fsqa-pipeline/phase1-eval-gen.ipynb", "fsqa-pipeline/phase5-testset-rebuild.ipynb"]
# Guarded in part A.
GUARDED_NOW = [
    "benchmarks/allergen-bench-v1.ipynb", "benchmarks/allergen-bench-v101.ipynb", "benchmarks/bakeoff-r3-legacy.ipynb",
    "benchmarks/bakeoff-r4-effort.ipynb", "benchmarks/bakeoff-r6-nemotron.ipynb",
    "benchmarks/probe-sonnet46-factcheck-leg.ipynb", "benchmarks/recall-bench-2-0-3.ipynb",
    "benchmarks/recall-bench-v2.ipynb", "benchmarks/recall-bench-v201.ipynb", "benchmarks/recall-bench-v202.ipynb",
    "drb/drb-6-research-pro-runner.ipynb", "drb/drb-deep-investigation-grading.ipynb",
    "drb/drb-deep-investigation-part-2.ipynb", "drb/drb-quick-scan-grading.ipynb", "drb/drb-quick-scan.ipynb",
    "drb/drb-research-pro.ipynb", "fsqa-pipeline/phase1-eval-gen-v2.ipynb", "fsqa-pipeline/phase1-eval-gen-v3.ipynb",
    "fsqa-pipeline/phase2-model-eval.ipynb", "fsqa-pipeline/phase2b-screen.ipynb", "fsqa-pipeline/phase2c-confirm.ipynb",
    "fsqa-pipeline/phase3-generation.ipynb", "fsqa-pipeline/phase4-training-pairs.ipynb",
]
# Setup cells that make data directories (not run or ledger directories) before RUN_ID is set, and what they make
# under the working directory. Left as they are: item 2 covers run and ledger directories only.
# RUN_ID path builds in the notebooks left out of the public copy. Where one is absent, its builds are not expected.
EXCLUDED_BUILDS = {"benchmarks/probe-sonnet46-factcheck-leg.ipynb": 1, "drb/drb-6-research-pro-runner.ipynb": 1}
EARLY_DATA_DIRS = {"benchmarks/recall-bench-v202.ipynb": ["work/local-data/v2", "work/local-data/v2/judged",
                                                          "work/local-data/v2/results", "work/local-data/v2/runs"]}

PATH_CALLS = {"os.path.join", "pathlib.Path", "Path", "pathlib.PurePath", "PurePath"}
MAKES = {"makedirs", "mkdir", "open"}   # os.makedirs, Path.mkdir, os.open (the runner lock)


def _live(src):
    return "\n".join(l if not l.startswith(("!", "%")) else "" for l in src.splitlines())


def _code_cells(nb_path):
    """[(raw cell index, parsed module)] for every code cell, in notebook order."""
    obj = json.load(open(REPO / nb_path, encoding="utf-8"))
    return [(i, ast.parse(_live("".join(c["source"])))) for i, c in enumerate(obj["cells"]) if c["cell_type"] == "code"]


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _aliases(cells):
    """RUN_ID plus every name assigned straight from it at top level (run_id = RUN_ID)."""
    out = {"RUN_ID"}
    for _, tree in cells:
        for s in tree.body:
            if isinstance(s, ast.Assign) and isinstance(s.value, ast.Name) and s.value.id == "RUN_ID":
                out |= {t.id for t in s.targets if isinstance(t, ast.Name)}
    return out


def _path_builds(stmt, names):
    """Path expressions inside one statement that use any of names: os.path.join / Path calls, '/' and '+' joins, and
    f-strings with a slash in their text."""
    out = []
    for n in ast.walk(stmt):
        if isinstance(n, ast.Call) and ast.unparse(n.func) in PATH_CALLS:
            if any(_names(a) & names for a in n.args):
                out.append(n)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.Add)):
            if (_names(n.left) | _names(n.right)) & names and (isinstance(n.op, ast.Div) or "/" in ast.unparse(n)):
                out.append(n)
        elif isinstance(n, ast.JoinedStr):
            text = "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            used = set().union(*[_names(v) for v in n.values if isinstance(v, ast.FormattedValue)] or [set()])
            if ("/" in text or "\\" in text) and used & names:
                out.append(n)
    return out


def _is_standard_guard(s):
    return (isinstance(s, ast.Assign) and [ast.unparse(t) for t in s.targets] == ["RUN_ID"]
            and ast.unparse(s.value) == "_checked_run_id(RUN_ID)")


def _is_regex_guard(s):
    return (isinstance(s, ast.If) and "re.fullmatch" in ast.unparse(s.test) and "RUN_ID" in _names(s.test)
            and len(s.body) == 1 and isinstance(s.body[0], ast.Raise)
            and ast.unparse(s.body[0].exc.func) == "RuntimeError")


def _assigns_run_id(s):
    targets = []
    for n in ast.walk(s):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets += n.targets if isinstance(n, ast.Assign) else [n.target]
    return any(isinstance(t, ast.Name) and t.id == "RUN_ID" for t in targets)


def _standard_helper():
    for _, tree in _code_cells(COMPLICITY):
        defs = [s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == "_checked_run_id"]
        if defs:
            return defs[0]
    raise AssertionError("complicity helper not found")


class RunIdPathJoins(unittest.TestCase):
    """Static check over every notebook in the repository."""

    def _inventory(self):
        """{notebook: [(cell, line, source of the path expression)]} for every RUN_ID path build."""
        inv = {}
        for nb in NOTEBOOKS:
            cells = _code_cells(nb)
            names = _aliases(cells)
            for ci, tree in cells:
                for s in tree.body:
                    for n in _path_builds(s, names):
                        inv.setdefault(nb, []).append((ci, n.lineno, ast.unparse(n)))
        return inv

    def test_the_notebooks_that_build_run_id_paths_are_the_expected_ones(self):
        inv = self._inventory()
        self.assertEqual(sorted(inv), sorted(present(STANDARD_BEFORE + REGEX_LEFT + GUARDED_NOW)),
                         "a notebook that builds a path from RUN_ID must be added to this test on purpose")
        gone = sum(n for nb, n in EXCLUDED_BUILDS.items() if absent(nb))
        self.assertEqual(sum(len(v) for v in inv.values()), 45 - gone, "36 direct joins and 9 through run_id = RUN_ID")

    def test_every_run_id_path_build_comes_after_the_check(self):
        for nb in STANDARD_BEFORE + REGEX_LEFT + GUARDED_NOW:
            with self.subTest(nb=nb):
                skip_if_absent(self, nb)
                cells = _code_cells(nb)
                names = _aliases(cells)
                is_guard = _is_regex_guard if nb in REGEX_LEFT else _is_standard_guard
                guards = [(k, j) for k, (_, tree) in enumerate(cells) for j, s in enumerate(tree.body) if is_guard(s)]
                self.assertEqual(len(guards), 1, "exactly one top-level RUN_ID check")
                g = guards[0]
                for k, (ci, tree) in enumerate(cells):
                    for j, s in enumerate(tree.body):
                        for n in _path_builds(s, names):
                            self.assertGreater((k, j), g, f"cell {ci} line {n.lineno}: {ast.unparse(n)} comes before "
                                                          "the RUN_ID check")
                        if (k, j) > g:
                            self.assertFalse(_assigns_run_id(s), f"cell {ci}: RUN_ID changes after the check")

    def test_helper_is_the_complicity_helper_verbatim(self):
        want = ast.dump(_standard_helper())
        for nb in STANDARD_BEFORE + GUARDED_NOW:
            with self.subTest(nb=nb):
                skip_if_absent(self, nb)
                defs = [s for _, tree in _code_cells(nb) for s in tree.body
                        if isinstance(s, ast.FunctionDef) and s.name == "_checked_run_id"]
                self.assertEqual(len(defs), 1)
                self.assertEqual(ast.dump(defs[0]), want, "the helper must be copied verbatim, docstring included")

    def test_setup_cell_makes_no_run_or_ledger_directory_and_no_lock_before_the_check(self):
        for nb in STANDARD_BEFORE + REGEX_LEFT + GUARDED_NOW:
            with self.subTest(nb=nb):
                skip_if_absent(self, nb)
                cells = _code_cells(nb)
                is_guard = _is_regex_guard if nb in REGEX_LEFT else _is_standard_guard
                ci, tree = [(ci, t) for ci, t in cells if any(is_guard(s) for s in t.body)][0]
                early = []
                for s in tree.body:
                    if is_guard(s):
                        break
                    if isinstance(s, (ast.FunctionDef, ast.ClassDef)):
                        continue
                    for n in ast.walk(s):
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MAKES:
                            if n.func.attr == "open" and ast.unparse(n.func) != "os.open":
                                continue
                            early.append(ast.unparse(n))
                self.assertEqual(len(early), 1 if nb in EARLY_DATA_DIRS else 0, early)


class RegexChecksLeftInPlace(unittest.TestCase):
    """The four kept regex checks refuse every escaping value, and refuse some values the helper accepts. That is why
    they were kept and not replaced by the helper. (This passes on the base too: it records why the regexes stay.)"""

    ESCAPES = ["/etc/run", "/abs/run", "../escape", "a/../../escape", "..\\escape", "..", "run/../..", "   ",
               "\\\\server\\share", "\\rooted"]
    GOOD = ["resume-20261107", "run_1", "a.b", "20261107T082247Z"]

    def _check(self, nb):
        stmts = [s for _, tree in _code_cells(nb) for s in tree.body if _is_regex_guard(s)]
        self.assertEqual(len(stmts), 1)
        code = compile(ast.Module(body=stmts, type_ignores=[]), "<check>", "exec")

        def run(value):
            exec(code, {"re": re, "RUN_ID": value})
        return run

    def test_regex_checks_refuse_escapes_and_are_stricter_than_the_helper(self):
        ns = {"os": os}
        exec(compile(ast.Module(body=[_standard_helper()], type_ignores=[]), "<helper>", "exec"), ns)
        helper = ns["_checked_run_id"]
        for nb in REGEX_LEFT:
            run = self._check(nb)
            for bad in self.ESCAPES:
                with self.subTest(nb=nb, bad=bad):
                    self.assertRaises(RuntimeError, run, bad)
            for good in self.GOOD:
                with self.subTest(nb=nb, good=good):
                    run(good)
            with self.subTest(nb=nb, stricter="a/b"):
                self.assertEqual(helper("a/b"), "a/b")
                self.assertRaises(RuntimeError, run, "a/b")


RUN_ID_NONE = re.compile(r"^RUN_ID = None", re.M)


def _imports_before(obj, raw_index):
    """Top-level standard-library import statements of the code cells before the setup cell, as source lines. The
    setup cells rely on them (os, datetime, ...); the cells themselves also run pip installs and downloads, so they are
    not run. Third-party imports are left out: the harness stubs the network clients, and the rest are not needed."""
    lines = []
    for c in obj["cells"][:raw_index]:
        if c["cell_type"] != "code":
            continue
        for s in ast.parse(_live("".join(c["source"]))).body:
            if isinstance(s, ast.Import) and all(a.name.split(".")[0] in sys.stdlib_module_names for a in s.names):
                lines.append(ast.unparse(s))
            elif isinstance(s, ast.ImportFrom) and s.level == 0 and s.module.split(".")[0] in sys.stdlib_module_names:
                lines.append(ast.unparse(s))
    return lines


# Setup cells that reach the RUN_ID check when run alone. Left out, because an earlier check in the same cell refuses
# first whatever RUN_ID is: bakeoff-r3-legacy and recall-bench-v201 (no confirmed price) and the two DRB grading
# notebooks (GRADER_COST_BOUNDED). The static tests above cover them.
BEHAVIOR = [nb for nb in GUARDED_NOW if nb not in (
    "benchmarks/bakeoff-r3-legacy.ipynb", "benchmarks/recall-bench-v201.ipynb",
    "drb/drb-deep-investigation-grading.ipynb", "drb/drb-quick-scan-grading.ipynb")]


def _refused_setup_run(nb, literal, td):
    """Run the notebook's setup cell with RUN_ID = literal, cut after the first LEDGER_PATH line that follows the last
    RUN_ID join, so the runner lock, smoke calls and later cells do not run on a base that has no check."""
    obj = json.load(open(REPO / nb, encoding="utf-8"))
    code = [i for i, c in enumerate(obj["cells"]) if c["cell_type"] == "code"]
    raw = [i for i in code if RUN_ID_NONE.search("".join(obj["cells"][i]["source"]))]
    assert len(raw) == 1, (nb, raw)
    ri = raw[0]
    text = "".join(obj["cells"][ri]["source"])
    assert len(RUN_ID_NONE.findall(text)) == 1, nb
    text = RUN_ID_NONE.sub(f"RUN_ID = {literal}", text, count=1)
    lines = text.splitlines(keepends=True)
    last_join = max(i for i, l in enumerate(lines) if re.search(r"os\.path\.join\(.*\bRUN_ID\)", l))
    cut = next(i for i in range(last_join, len(lines)) if lines[i].startswith("LEDGER_PATH = "))
    text = "".join(line + "\n" for line in _imports_before(obj, ri)) + "".join(lines[:cut + 1])
    obj["cells"][ri]["source"] = text.splitlines(keepends=True)
    tmp_nb = os.path.join(td, "setup.ipynb")
    with open(tmp_nb, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    work = os.path.join(td, "work")
    os.makedirs(os.path.join(work, "local-data"))   # phase2b refuses unless its volume path exists
    ledger = os.path.join(td, "ledger")
    before = _tree(td)
    ns, rec, exc = nbharness.run_cells(tmp_nb, [code.index(ri)], cwd=work,
                                       env={"SPEND_LEDGER_DIR": ledger, "OPENROUTER_API_KEY": "test",
                                            "OPENAI_API_KEY": "test", "BENCHMARK_API_KEY": "test"})
    return rec, exc, sorted(_tree(td) - before), ledger


def _tree(root):
    return {os.path.relpath(os.path.join(r, n), root).replace(os.sep, "/") for r, ds, fs in os.walk(root) for n in ds + fs}


class RunIdRefusal(unittest.TestCase):
    """The setup cell of each newly guarded notebook refuses a RUN_ID that would leave the ledger root, and a falsy
    RUN_ID ("", 0, False: only None starts a new run), before it makes the ledger root, a run directory or the runner
    lock, and before anything is sent."""

    def test_newly_guarded_setup_cells_refuse_an_escaping_or_falsy_run_id(self):
        self.assertEqual(len(BEHAVIOR), 19)
        for nb in BEHAVIOR:
            for literal in ['"../escape"', '"a/../../escape"', '""', "0", "False"]:
                with self.subTest(nb=nb, literal=literal), tempfile.TemporaryDirectory() as td:
                    skip_if_absent(self, nb)
                    rec, exc, made, ledger = _refused_setup_run(nb, literal, td)
                    self.assertIsInstance(exc, RuntimeError, repr(exc))
                    self.assertIn("RUN_ID", str(exc))
                    self.assertFalse(os.path.exists(ledger), "no ledger root may be made")
                    self.assertFalse(any("escape" in p for p in made), made)
                    self.assertFalse(any(p.endswith("runner.lock") for p in made), made)
                    self.assertEqual(made, EARLY_DATA_DIRS.get(nb, []), "nothing else may be made")
                    self.assertEqual(rec.paid, [])


    def test_regex_checked_setup_cells_refuse_a_falsy_run_id(self):
        # Only None starts a new run here too: "", 0 and False stop at the non-empty-string check that runs before the
        # kept regex check, with nothing created and nothing sent.
        for nb in REGEX_LEFT:
            for literal in ['""', "0", "False"]:
                with self.subTest(nb=nb, literal=literal), tempfile.TemporaryDirectory() as td:
                    rec, exc, made, ledger = _refused_setup_run(nb, literal, td)
                    self.assertIsInstance(exc, RuntimeError, repr(exc))
                    self.assertIn("RUN_ID must be a non-empty string", str(exc))
                    self.assertFalse(os.path.exists(ledger), "no ledger root may be made")
                    self.assertEqual(made, [], "nothing may be made")
                    self.assertEqual(rec.paid, [])


DRB_URL = "https://github.com/Ayanami0730/deep_research_bench.git"
GRADING = ["drb/drb-deep-investigation-grading.ipynb", "drb/drb-deep-investigation-part-2.ipynb",
           "drb/drb-quick-scan-grading.ipynb"]


DRB_SITES = [("drb/drb-6-research-pro-runner.ipynb", 2), ("drb/drb-deep-investigation-part-2.ipynb", 2),
             ("drb/drb-deep-investigation-part-2.ipynb", 4), ("drb/drb-quick-scan.ipynb", 2),
             ("drb/drb-research-pro.ipynb", 1)]


def _clone_block(nb, raw_cell):
    """The statements between QUERY_FILE = ... and queries = ... in one clone cell (prints left out), as source.
    That is the clone, the pinned checkout and the HEAD check, or on the base the old QUERY_FILE-gated clone."""
    obj = json.load(open(REPO / nb, encoding="utf-8"))
    body = ast.parse(_live("".join(obj["cells"][raw_cell]["source"]))).body
    names = [ast.unparse(s.targets[0]) if isinstance(s, ast.Assign) else None for s in body]
    a, b = names.index("QUERY_FILE"), names.index("queries")
    keep = [s for s in body[a + 1:b] if not (isinstance(s, ast.Expr) and ast.unparse(s.value.func) == "print")]
    return "\n".join(ast.unparse(s) for s in keep) + "\n"


class GraderClonePins(unittest.TestCase):
    """deep_research_bench is cloned only when it is missing, the fresh clone is checked out at the grading notebooks'
    GRADER_REVISION, and on every run, fresh clone or not, the cell refuses to read the queries unless HEAD is at that
    commit. An existing checkout is never checked out, reset or changed."""

    def _revision(self):
        revs = set()
        for nb in GRADING:
            for _, tree in _code_cells(nb):
                for s in tree.body:
                    if isinstance(s, ast.Assign) and [ast.unparse(t) for t in s.targets] == ["GRADER_REVISION"]:
                        revs.add(ast.literal_eval(s.value))
        self.assertEqual(len(revs), 1, revs)
        rev = revs.pop()
        self.assertRegex(rev, r"^[0-9a-f]{40}$")
        return rev

    def test_every_drb_clone_is_pinned_to_the_grading_revision(self):
        rev = self._revision()
        want = ["git", "-C", "deep_research_bench", "checkout", "-q", "--detach", rev]
        clones = 0
        for nb in NOTEBOOKS:
            for ci, tree in _code_cells(nb):
                for parent in ast.walk(tree):
                    body = getattr(parent, "body", None)
                    if not isinstance(body, list):
                        continue
                    for i, s in enumerate(body):
                        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                and ast.unparse(s.value.func) == "subprocess.run" and s.value.args
                                and isinstance(s.value.args[0], ast.List)):
                            continue
                        argv = [ast.literal_eval(a) for a in s.value.args[0].elts if isinstance(a, ast.Constant)]
                        if argv[:2] != ["git", "clone"] or DRB_URL not in argv:
                            continue
                        clones += 1
                        with self.subTest(nb=nb, cell=ci, line=s.lineno):
                            self.assertLess(i + 1, len(body), "the clone must be followed by the pinned checkout")
                            nxt = body[i + 1]
                            self.assertEqual(ast.unparse(nxt.value.func) if isinstance(nxt, ast.Expr) and
                                             isinstance(nxt.value, ast.Call) else None, "subprocess.run")
                            pins = {ast.unparse(t): ast.literal_eval(a.value) for a in tree.body
                                    if isinstance(a, ast.Assign) and isinstance(a.value, ast.Constant)
                                    for t in a.targets if ast.unparse(t) == "_DRB_PIN"}
                            argv2 = [pins[ast.unparse(e)] if isinstance(e, ast.Name) else ast.literal_eval(e)
                                     for e in nxt.value.args[0].elts]
                            self.assertEqual(argv2, want)
                            self.assertEqual([(k.arg, ast.literal_eval(k.value)) for k in nxt.value.keywords],
                                             [("check", True)])
        gone = 1 if absent("drb/drb-6-research-pro-runner.ipynb") else 0   # the runner clones once
        self.assertEqual(clones, 5 - gone, "four notebooks clone deep_research_bench, part 2 twice")

    def test_runner_setup_cells_clone_then_check_out_the_pinned_commit(self):
        # The setup cells of these three notebooks clone the query set when it is missing. Run each with no query
        # file: the stubbed clone writes nothing, so the cell stops at open(QUERY_FILE). The subprocess calls it made
        # must be the clone and then the pinned checkout. (Part 2 clones only after the research-submit refusal.)
        rev = self._revision()
        for nb in ["drb/drb-6-research-pro-runner.ipynb", "drb/drb-quick-scan.ipynb", "drb/drb-research-pro.ipynb"]:
            with self.subTest(nb=nb), tempfile.TemporaryDirectory() as td:
                skip_if_absent(self, nb)
                idx = [k for k, src in enumerate(nbharness.load_cells(nb)[0]) if RUN_ID_NONE.search(src)][0]
                work = os.path.join(td, "work")
                os.makedirs(work)
                ns, rec, exc = nbharness.run_cells(nb, [idx], cwd=work, env={
                    "SPEND_LEDGER_DIR": os.path.join(td, "ledger"), "BENCHMARK_API_KEY": "test"},
                    pre_exec=lambda ns, rec: setattr(rec, "git_head", rev))
                self.assertIsInstance(exc, FileNotFoundError, repr(exc))
                self.assertIn("query.jsonl", str(exc))
                self.assertEqual(rec.subprocess, [["git", "clone", DRB_URL],
                                                  ["git", "-C", "deep_research_bench", "checkout", "-q", "--detach", rev],
                                                  ["git", "-C", "deep_research_bench", "rev-parse", "HEAD"]])

    def _run_block(self, nb, raw_cell, td, existing, head):
        """Run one site's clone block in the harness. existing: make deep_research_bench (with a query file) first.
        head: what the stubbed git rev-parse HEAD reports."""
        work = os.path.join(td, "work")
        os.makedirs(work)
        if existing:
            q = pathlib.Path(work, "deep_research_bench", "data", "prompt_data", "query.jsonl")
            q.parent.mkdir(parents=True)
            q.write_text('{"id": "q1", "prompt": "p"}\n')
        src = ('import subprocess\nfrom pathlib import Path\n'
               'QUERY_FILE = "deep_research_bench/data/prompt_data/query.jsonl"\n' + _clone_block(nb, raw_cell))
        tmp_nb = os.path.join(td, "block.ipynb")
        with open(tmp_nb, "w", encoding="utf-8") as f:
            json.dump({"cells": [{"cell_type": "code", "source": src.splitlines(keepends=True)}]}, f)
        return nbharness.run_cells(tmp_nb, [0], cwd=work, pre_exec=lambda ns, rec: setattr(rec, "git_head", head))

    def test_each_clone_site_refuses_an_existing_checkout_at_another_commit_and_leaves_it_alone(self):
        rev = self._revision()
        head_cmd = ["git", "-C", "deep_research_bench", "rev-parse", "HEAD"]
        for nb, raw_cell in DRB_SITES:
            with self.subTest(nb=nb, cell=raw_cell, case="existing, wrong HEAD"), tempfile.TemporaryDirectory() as td:
                skip_if_absent(self, nb)
                ns, rec, exc = self._run_block(nb, raw_cell, td, existing=True, head="0" * 40)
                self.assertIsInstance(exc, RuntimeError, repr(exc))
                self.assertIn(rev, str(exc))
                self.assertIn(f"git -C deep_research_bench checkout --detach {rev}", str(exc), "the manual fix is printed")
                self.assertEqual(rec.subprocess, [head_cmd], "read-only: no clone, checkout or reset")
                self.assertEqual(rec.paid, [])
            with self.subTest(nb=nb, cell=raw_cell, case="existing, pinned HEAD"), tempfile.TemporaryDirectory() as td:
                skip_if_absent(self, nb)
                ns, rec, exc = self._run_block(nb, raw_cell, td, existing=True, head=rev)
                self.assertIsNone(exc, repr(exc))
                self.assertEqual(rec.subprocess, [head_cmd])
                self.assertEqual(rec.paid, [])
            with self.subTest(nb=nb, cell=raw_cell, case="missing"), tempfile.TemporaryDirectory() as td:
                skip_if_absent(self, nb)
                ns, rec, exc = self._run_block(nb, raw_cell, td, existing=False, head=rev)
                self.assertIsNone(exc, repr(exc))
                self.assertEqual(rec.subprocess, [["git", "clone", DRB_URL],
                                                  ["git", "-C", "deep_research_bench", "checkout", "-q", "--detach", rev],
                                                  head_cmd])

if __name__ == "__main__":
    unittest.main()
