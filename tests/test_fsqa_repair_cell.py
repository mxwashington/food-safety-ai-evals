"""Offline tests for the optional repair cell of fsqa-pipeline/phase1-eval-gen-v3.ipynb.

The repair cell REWRITES questions.jsonl in the run folder named by RUN. These tests run the cell's own source in a
disposable temp folder (the working directory is moved there, so ./local-data resolves inside it) with RUN set to
each case. A RUN that does not name exactly one existing folder inside ./local-data/phase1_fsqa_eval must be refused
with RuntimeError before any file is written. Standard library only. No network.
"""
import contextlib, io, json, os, pathlib, tempfile, unittest, warnings

REPO = pathlib.Path(__file__).resolve().parent.parent
NB = REPO / "fsqa-pipeline" / "phase1-eval-gen-v3.ipynb"
RUN_LINE = 'RUN = "<YOUR_RUN_ID>"'

KEPT = [{"id": "a", "domain": "gmp", "difficulty": "Straightforward", "seed_section": "117.3",
         "citation": "21 CFR 117.3", "reference_answer": "Defined in 21 CFR 117.3."}]
REJ = [{"id": "b", "domain": "gmp", "difficulty": "requires application", "seed_section": "117.5", "citation": "",
        "reference_answer": "See 21 CFR 117.5(a).", "_reject_reason": "no citation"}]


def repair_source():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    hits = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and RUN_LINE in "".join(c["source"])]
    assert len(hits) == 1, f"expected one repair cell, found {len(hits)}"
    assert hits[0].count(RUN_LINE) == 1
    return hits[0]


def plant(folder):
    """Write a questions.jsonl and rejected.jsonl pair into folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in KEPT))
    (folder / "rejected.jsonl").write_text("".join(json.dumps(q) + "\n" for q in REJ))


def snapshot(top):
    """{relative path: bytes} for every file under top, symlinks not followed."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for n in filenames:
            p = os.path.join(dirpath, n)
            with open(p, "rb") as f:
                out[os.path.relpath(p, top)] = f.read()
    return out


class RepairCellRunCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="repaircell_")
        self.top = pathlib.Path(self._tmp.name)
        self.root = self.top / "local-data" / "phase1_fsqa_eval"
        plant(self.root / "run1")                  # the normal run folder
        plant(self.root / "run1" / "sub")          # reachable only with a separator
        plant(self.root / "<YOUR_RUN_ID>")         # a folder literally named like the placeholder
        plant(self.root)                           # the root itself, reached by a blank RUN or "."
        plant(self.top / "outside")                # outside the root
        (self.root / "afile").write_text("not a folder\n")
        self.src = repair_source()

    def tearDown(self):
        self._tmp.cleanup()

    def run_cell(self, run_value):
        src = self.src.replace(RUN_LINE, f"RUN = {run_value!r}", 1)
        old = os.getcwd()
        os.chdir(self.top)
        try:
            # The cell's reads use bare open() calls, which is the cell's own style; silence that warning only.
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                exec(compile(src, "repair_cell", "exec"), {"__name__": "__main__"})
        finally:
            os.chdir(old)

    def assert_refused(self, run_value):
        before = snapshot(self.top)
        with self.assertRaises(RuntimeError) as cm:
            self.run_cell(run_value)
        self.assertIn("Refusing to repair", str(cm.exception))
        self.assertEqual(snapshot(self.top), before, f"a file changed for RUN={run_value!r}")

    def test_accepts_normal_folder_and_rewrites_questions(self):
        before = snapshot(self.top)
        self.run_cell("run1")
        after = snapshot(self.top)
        changed = {k for k in before if before[k] != after.get(k)} | (set(after) - set(before))
        self.assertEqual(changed, {os.path.join("local-data", "phase1_fsqa_eval", "run1", "questions.jsonl")})
        rows = [json.loads(l) for l in (self.root / "run1" / "questions.jsonl").read_text().splitlines()]
        self.assertEqual([q["id"] for q in rows], ["a", "b"])
        self.assertEqual([q["difficulty"] for q in rows], ["recall", "application"])
        self.assertEqual(rows[1]["citation"], "21 CFR 117.5(a)")
        self.assertNotIn("_reject_reason", rows[1])

    def test_refuses_placeholder(self):
        self.assert_refused("<YOUR_RUN_ID>")

    def test_refuses_blank(self):
        for v in ["", "   "]:
            with self.subTest(RUN=v):
                self.assert_refused(v)

    def test_refuses_non_string(self):
        for v in [None, 0, False]:
            with self.subTest(RUN=v):
                self.assert_refused(v)

    def test_refuses_absolute_path(self):
        self.assert_refused(str(self.top / "outside"))

    def test_refuses_dotdot_traversal(self):
        for v in ["../../outside", ".."]:
            with self.subTest(RUN=v):
                self.assert_refused(v)

    def test_refuses_path_separator(self):
        for v in ["run1/sub", "run1\\sub"]:
            with self.subTest(RUN=v):
                self.assert_refused(v)

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlink support")
    def test_refuses_symlink_pointing_outside_root(self):
        os.symlink(self.top / "outside", self.root / "escape", target_is_directory=True)
        self.assert_refused("escape")

    def test_refuses_target_that_is_not_an_existing_folder_inside_root(self):
        for v in ["missing", "afile", "."]:
            with self.subTest(RUN=v):
                self.assert_refused(v)


if __name__ == "__main__":
    unittest.main()
