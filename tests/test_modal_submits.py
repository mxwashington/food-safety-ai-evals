"""Modal research-job submits refuse until a hosted per-job cost bound is verified. Mocked only: requests is a
recorder, sleeps are no-ops, no network, no paid call. Covers drb-6-research-pro-runner (smoke cell and full-run cell),
drb-quick-scan, drb-research-pro and the three submit cells of drb-deep-investigation-part-2 (smoke cell and the two
query-run cells). Status polls are not paid and must stay available.

With the shipped constants every submit entry raises the refusal with rec.paid empty and no ledger line. With
RESEARCH_COST_BOUNDED patched to True in the cell text, the submit reaches reserve_job and the stubbed post as before.
"""
import json, os, pathlib, re, sys, tempfile, time, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nbharness  # noqa: E402
from public_copy_exclusions import skip_if_absent  # noqa: E402

RUNNER = "drb/drb-6-research-pro-runner.ipynb"
QUICK = "drb/drb-quick-scan.ipynb"
PRO = "drb/drb-research-pro.ipynb"
PART2 = "drb/drb-deep-investigation-part-2.ipynb"
QFILE = "deep_research_bench/data/prompt_data/query.jsonl"
# The commit the DRB clone sites check before reading QFILE (_DRB_PIN in each notebook, GRADER_REVISION in the grading
# notebooks). The stubbed checkout reports it, so these tests reach the submit code as before.
DRB_PIN = "f2735b5c3636759c22f9e936c0232de8bf545b67"
MARK = "Refusing to submit"
FORBIDDEN = ("SPEND_LEDGER_DIR", "RUN_BUDGET_USD", "JOB_WORST_CASE_USD", "runner.lock")

# Prelude put in front of the first executed cell: stubbed requests.post/get and no-op sleeps.
PRELUDE = ("import requests as _rq, time as _tm\n_rq.post = _TEST_POST\n_rq.get = _TEST_GET\n_tm.sleep = lambda s: None\n")


class Resp:
    status_code = 200

    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


def make_net(rec):
    def post(*a, **k):
        rec.note_paid("requests.post")
        return Resp({"job_id": "j1", "results": {"report": "ARTICLE"}, "meta": {}})

    def get(*a, **k):
        rec.other.append("requests.get")
        return Resp({"status": "complete", "results": {"report": "ARTICLE"}, "meta": {}})
    return post, get


def run(nb, ids, bounded=False, cut_smoke=False):
    cells, _ = nbharness.load_cells(nb)
    cells = list(cells)
    if cut_smoke:
        # part-2 cell 0 holds the guard definitions, the ledger and a smoke submit. Keep only the definitions so the
        # query-run cells (which call require_bounded_research() and reserve_job()) can be run on their own.
        head, sep, _tail = cells[0].partition("\n# Quick smoke test")
        assert sep, "smoke marker not found in part-2 cell 0"
        cells[0] = head
    first = ids[0]
    cells[first] = PRELUDE + cells[first]
    if bounded:
        n = 0
        for i in ids:
            if "RESEARCH_COST_BOUNDED = False" in cells[i]:
                cells[i] = cells[i].replace("RESEARCH_COST_BOUNDED = False", "RESEARCH_COST_BOUNDED = True")
                n += 1
        assert n == 1
    cwd = tempfile.mkdtemp(prefix="nbmodal_")

    def pre(ns, rec):
        p = pathlib.Path(QFILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"id": "q1", "prompt": "What is FSMA 204?"}) + "\n")
        rec.git_head = DRB_PIN
        ns["_TEST_POST"], ns["_TEST_GET"] = make_net(rec)

    with mock.patch.object(nbharness, "load_cells", lambda p: (cells, None)), mock.patch.object(time, "sleep", lambda s: None):
        ns, rec, exc = nbharness.run_cells(nb, ids, env={"BENCHMARK_API_KEY": "test-key"}, cwd=cwd, pre_exec=pre)
    return ns, rec, exc, cwd


def ledger_lines(ns):
    p = ns.get("LEDGER_PATH")
    if not p or not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


# (notebook, code cells to run, label)
ENTRIES = [
    (RUNNER, [1, 2, 3], "runner smoke"),
    (RUNNER, [1, 2, 4], "runner full run"),
    (QUICK, [1], "quick scan"),
    (PRO, [0], "research pro"),
    (PART2, [0], "part-2 smoke"),
]
# part-2 query-run cells: cell 0 is run first with its smoke submit cut off (definitions and ledger only).
PART2_RUNS = [[0, 1], [0, 3]]


class Refusal(unittest.TestCase):
    def test_every_submit_entry_refuses(self):
        for nb, ids, label in ENTRIES:
            with self.subTest(label):
                skip_if_absent(self, nb)
                ns, rec, exc, cwd = run(nb, ids)
                self.assertIsInstance(exc, RuntimeError, label)
                self.assertTrue(str(exc).startswith(MARK), str(exc))
                self.assertEqual(rec.paid, [], label)
                self.assertEqual(ledger_lines(ns), [], label)
                self.assertFalse(any(os.path.exists(os.path.join(cwd, f)) for f in
                                     ("drb_v6_research_pro_results.jsonl", "drb_quick_scan_results.jsonl", "drb_v5_research_pro_results.jsonl")))
                self.assertIs(ns["RESEARCH_COST_BOUNDED"], False)

    def test_message_explains_and_avoids_guard_names(self):
        ns, rec, exc, _ = run(QUICK, [1])
        m = str(exc)
        for phrase in ("flat per-submit reservation", "cost ceiling", "cancellation", "polling timeout", "stay off"):
            self.assertIn(phrase, m)
        for bad in FORBIDDEN:
            self.assertNotIn(bad, m)

    def test_truthy_non_true_value_still_refuses(self):
        cells, _ = nbharness.load_cells(QUICK)
        src = cells[1].replace("RESEARCH_COST_BOUNDED = False", 'RESEARCH_COST_BOUNDED = "yes"')
        self.assertNotEqual(src, cells[1])
        ns = {}
        exec(compile(src.split("# --- Spend guard")[0], "x", "exec"), ns)
        with self.assertRaises(RuntimeError):
            ns["require_bounded_research"]()

    def test_part2_query_run_cells_refuse(self):
        for ids in PART2_RUNS:
            with self.subTest(ids):
                ns, rec, exc, cwd = run(PART2, ids, cut_smoke=True)
                self.assertIsInstance(exc, RuntimeError, ids)
                self.assertTrue(str(exc).startswith(MARK), str(exc))
                self.assertEqual(rec.paid, [], ids)
                self.assertEqual(ledger_lines(ns), [], ids)
                self.assertFalse(os.path.exists(os.path.join(cwd, "drb_v4_deep_results.jsonl")))
                self.assertIs(ns["RESEARCH_COST_BOUNDED"], False)

    def test_part2_cut_check_is_effective(self):
        """The cut leaves the definitions: if the cut text still contained the smoke guard, cell 0 would refuse and the
        query-run cells would never run, so the refusal tests above would pass for the wrong reason."""
        cells, _ = nbharness.load_cells(PART2)
        head = cells[0].partition("\n# Quick smoke test")[0]
        self.assertIn("def require_bounded_research", head)
        self.assertIn("def reserve_job", head)
        self.assertIsNone(re.search(r"(?m)^require_bounded_research\(\)", head))
        ns, rec, exc, _ = run(PART2, [0], cut_smoke=True)
        self.assertIsNone(exc, repr(exc))
        self.assertEqual(rec.paid, [])


class Polls(unittest.TestCase):
    def check_poll(self, nb, ids):
        ns, rec, exc, _ = run(nb, ids)
        self.assertIsInstance(exc, RuntimeError)
        calls = []

        class R:
            @staticmethod
            def get(*a, **k):
                calls.append(1)
                return Resp({"status": "complete", "results": {"report": "ARTICLE"}})
        ns["requests"] = R
        with mock.patch.object(time, "sleep", lambda s: None):
            out = ns["poll_job"]("j1")
        self.assertEqual(calls, [1])
        self.assertTrue(out == "ARTICLE" or (isinstance(out, dict) and out["status"] == "complete"))
        self.assertEqual(rec.paid, [])

    def test_runner_poll_not_blocked(self):
        skip_if_absent(self, RUNNER)
        self.check_poll(RUNNER, [1, 2, 3])

    def test_research_pro_poll_not_blocked(self):
        self.check_poll(PRO, [0])


class Bounded(unittest.TestCase):
    """With the constant patched True the code behaves exactly as before: reserve, post, settle."""

    def check(self, nb, ids, outfile):
        ns, rec, exc, cwd = run(nb, ids, bounded=True)
        self.assertIsNone(exc, repr(exc))
        self.assertEqual(rec.paid, ["requests.post"])
        states = [(l["state"], l.get("purpose")) for l in ledger_lines(ns)]
        self.assertEqual([s for s, _ in states], ["reserved", "settled"])
        return cwd

    def test_runner_smoke(self):
        skip_if_absent(self, RUNNER)
        self.check(RUNNER, [1, 2, 3], None)

    def test_runner_full(self):
        skip_if_absent(self, RUNNER)
        cwd = self.check(RUNNER, [1, 2, 4], "drb_v6_research_pro_results.jsonl")
        self.assertTrue(os.path.exists(os.path.join(cwd, "drb_v6_research_pro_results.jsonl")))

    def test_quick_scan(self):
        cwd = self.check(QUICK, [1], None)
        self.assertTrue(os.path.exists(os.path.join(cwd, "drb_quick_scan_results.jsonl")))

    def test_research_pro(self):
        cwd = self.check(PRO, [0], None)
        self.assertTrue(os.path.exists(os.path.join(cwd, "drb_v5_research_pro_results.jsonl")))

    def test_part2_smoke(self):
        self.check(PART2, [0], None)

    def test_part2_query_run_cells(self):
        for ids in PART2_RUNS:
            with self.subTest(ids):
                ns, rec, exc, cwd = run(PART2, ids, bounded=True, cut_smoke=True)
                self.assertIsNone(exc, repr(exc))
                self.assertEqual(rec.paid, ["requests.post"])
                self.assertEqual([l["state"] for l in ledger_lines(ns)], ["reserved", "settled"])
                self.assertTrue(os.path.exists(os.path.join(cwd, "drb_v4_deep_results.jsonl")))


class NoUnguardedPost(unittest.TestCase):
    """Static check: every requests.post and every reserve_job call in these notebooks has a require_bounded_research()
    call earlier in the same cell, and the only cells that post are the known submit cells."""

    def test_guard_precedes_post_and_reserve(self):
        posting = {RUNNER: {3, 4}, QUICK: {1}, PRO: {0}, PART2: {0, 1, 3, 4}}
        for nb, expect in posting.items():
            with self.subTest(nb=nb):
                skip_if_absent(self, nb)
                cells, _ = nbharness.load_cells(nb)
                got = set()
                for i, src in enumerate(cells):
                    for m in re.finditer(r"requests\.post\(|=\s*reserve_job\(", src):
                        got.add(i)
                        before = src[:m.start()]
                        if nb == PART2 and i == 4:
                            # the grading cell: its reservation is the grader step, guarded by require_bounded_grading()
                            self.assertIn("require_bounded_grading()", before, f"{nb} cell {i}")
                            continue
                        self.assertIn("require_bounded_research()", before, f"{nb} cell {i}")
                        if "for attempt in" in src and nb == PART2:
                            # part-2 query-run cells: the guard is the first statement of the cell and is repeated
                            # right before the attempt loop.
                            self.assertTrue(src.lstrip().startswith("require_bounded_research()"), f"{nb} cell {i}")
                            self.assertRegex(src, r"\n    require_bounded_research\(\)\n    for attempt in", f"{nb} cell {i}")
                        elif "for attempt in" in src:
                            self.assertGreater(before.rfind("\n        require_bounded_research()"), before.rfind("for attempt in"),
                                               f"{nb} cell {i}: guard not at the top of the attempt")
                self.assertEqual(got, expect, nb)

    def test_no_other_post_paths(self):
        for nb in (RUNNER, QUICK, PRO, PART2):
            with self.subTest(nb=nb):
                skip_if_absent(self, nb)
                cells, _ = nbharness.load_cells(nb)
                for i, src in enumerate(cells):
                    for pat in (r"\.post\b", r"httpx", r"urlopen", r"\.request\(", r"\.submit\b", r"\.remote\(", r"\.spawn\("):
                        for m in re.finditer(pat, src):
                            line = src[src.rfind("\n", 0, m.start()) + 1: src.find("\n", m.start())]
                            ok = "requests.post(" in line or "fetch_secrets.remote(SECRET_NAMES)" in line
                            self.assertTrue(ok, f"{nb} cell {i}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
