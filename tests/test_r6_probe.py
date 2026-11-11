"""Provider-probe gate: benchmarks/bakeoff-r6-nemotron.ipynb must not run unless the provider probe succeeded and returned an
explicitly allowlisted provider. Mocked only: httpx is replaced by a recorder, no network, no paid call.

The probe cell is code cell 4 and the run cell is code cell 5. Extra snippets are injected between the notebook's own
cells (stub install, ledger snapshot, tampering) by patching nbharness.load_cells; nbharness.py itself is not edited.
"""
import contextlib, io, json, pathlib, sys, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nbharness  # noqa: E402

NB = "benchmarks/bakeoff-r6-nemotron.ipynb"
PROBE_TEXT = "Reply with the single word: ok"
NEMO = "nvidia/nemotron-3-ultra-550b-a55b"
OPUS = "anthropic/claude-opus-5"
MISSING = object()

STUBS = "httpx.post = _TEST_POST\nhttpx.get = _TEST_GET"
FAST = "RETRIES = 1"
LIMIT = 'PROMPTS = {"T1-benchmark": PROMPTS["T1-benchmark"]}'
MARK = ("_MARK = open(LEDGER_PATH).read() if os.path.exists(LEDGER_PATH) else ''\n"
        "_MARK_N = len(_ledger)")


class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._d, self.text = status, data, json.dumps(data) if data is not None else ""

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class Net:
    """Fake OpenRouter. `provider` is what the Nemotron responses report: a value, or MISSING to omit the key."""

    def __init__(self, rec, provider="DeepInfra", raise_for=None, status=200):
        self.rec, self.provider, self.raise_for, self.status = rec, provider, raise_for, status
        self.posts = []          # (is_probe, body)

    def post(self, url, headers=None, json=None, timeout=None, **k):
        body = json
        is_probe = body["messages"][0]["content"] == PROBE_TEXT
        self.posts.append((is_probe, body))
        self.rec.note_paid(f"httpx.post {body['model']} {'probe' if is_probe else 'RUN'}")
        if self.raise_for and body["model"] == self.raise_for:
            raise OSError("connection reset")
        if self.status != 200:
            return FakeResp(self.status, {"error": "boom"})
        d = {"choices": [{"message": {"content": "ok" if is_probe else "some words here"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001}}
        prov = self.provider if body["model"] == NEMO else "Anthropic"
        if prov is not MISSING:
            d["provider"] = prov
        return FakeResp(200, d)

    def get(self, url, **k):
        return types_ns(text="source text")

    @property
    def run_posts(self):
        return [b for p, b in self.posts if not p]

    @property
    def probe_posts(self):
        return [b for p, b in self.posts if p]


def types_ns(**k):
    import types
    return types.SimpleNamespace(**k)


def execute(plan, provider="DeepInfra", env=None, raise_for=None, status=200):
    """plan: list of ints (code cell indexes) and strs (injected code). Returns (ns, rec, exc, net, stdout)."""
    cells, _ = nbharness.load_cells(NB)
    extras = [x for x in plan if isinstance(x, str)]
    ids = [x if isinstance(x, int) else len(cells) + extras.index(x) for x in plan]
    holder = {}

    def pre(ns, rec):
        net = holder["net"] = Net(rec, provider, raise_for, status)
        ns["_TEST_POST"], ns["_TEST_GET"] = net.post, net.get

    out = io.StringIO()
    with mock.patch.object(nbharness, "load_cells", lambda p: (cells + extras, None)), contextlib.redirect_stdout(out):
        ns, rec, exc = nbharness.run_cells(NB, ids, env=dict({"OPENROUTER_API_KEY": "test-key"}, **(env or {})),
                                           pre_exec=pre)
    return ns, rec, exc, holder["net"], out.getvalue()


BASE = [0, STUBS, 1, FAST, 2, 3, 4]
RUN = [MARK, LIMIT, 5]


class R6Base(unittest.TestCase):
    def assert_refused(self, plan, *, probes=None, contains=None, **kw):
        ns, rec, exc, net, out = execute(plan, **kw)
        self.assertIsInstance(exc, RuntimeError, f"expected a refusal, got {exc!r}\n{out}")
        self.assertTrue(str(exc).startswith("Refusing to run: "), str(exc))
        for banned in ("SPEND_LEDGER_DIR", "RUN_BUDGET_USD", "JOB_WORST_CASE_USD", "runner.lock"):
            self.assertNotIn(banned, str(exc))
        if contains:
            self.assertIn(contains, str(exc))
        # no run request was built or sent, and nothing else paid either
        self.assertEqual(net.run_posts, [])
        self.assertEqual([p for p in rec.paid if not p.endswith("probe")], [])
        if probes is not None:
            self.assertEqual(len(net.probe_posts), probes)
        # no ledger reservation was written by the run cell
        if "_MARK" in ns:
            lp = pathlib.Path(ns["LEDGER_PATH"])
            on_disk = lp.read_text() if lp.exists() else ""
            self.assertEqual(on_disk, ns["_MARK"])
            self.assertEqual(len(ns["_ledger"]), ns["_MARK_N"])
        self.assertFalse(ns.get("jobs"), "run cell got as far as building jobs")
        return ns, rec, exc, net, out


class TestAllowed(R6Base):
    def test_allowed_provider_reaches_dispatcher_with_request_level_settings(self):
        ns, rec, exc, net, out = execute(BASE + RUN)
        self.assertIsNone(exc, out)
        self.assertEqual(ns["PROBE_RESULT"]["ok"], True)
        self.assertIs(ns["ok_to_run"], True)
        self.assertEqual(len(net.probe_posts), 2)
        runs = {b["model"]: b for b in net.run_posts}
        self.assertEqual(set(runs), {NEMO, OPUS})
        self.assertEqual(len(net.run_posts), 2)
        n = runs[NEMO]
        self.assertEqual(n["messages"], [{"role": "user", "content": ns["PROMPTS"]["T1-benchmark"]}])
        self.assertEqual({k: v for k, v in n.items() if k != "messages"}, {
            "model": NEMO, "max_tokens": 32000, "temperature": 1.0, "usage": {"include": True},
            "provider": {"only": ["deepinfra", "baseten"], "data_collection": "deny", "allow_fallbacks": False,
                         "max_price": {"prompt": 0.625, "completion": 3.125}}})
        o = runs[OPUS]
        self.assertEqual(o["reasoning"], {"effort": "low"})
        self.assertEqual(o["provider"], {"max_price": {"prompt": 10.0, "completion": 50.0}})
        # the reservation is written before each send and settled after
        lines = [json.loads(l) for l in open(ns["LEDGER_PATH"]) if l.strip()]
        self.assertEqual(sum(1 for l in lines if l["state"] == "reserved"), 4)
        self.assertEqual(sum(1 for l in lines if l["state"] == "settled"), 4)
        self.assertIn("All Nemotron calls stayed on allowed providers.", out)

    def test_case_and_whitespace_are_normalized(self):
        for prov in (" DeepInfra ", "BASETEN", "\tbaseten\n", "deepinfra", "  DEEPINFRA"):
            with self.subTest(prov=prov):
                ns, rec, exc, net, out = execute(BASE + RUN, provider=prov)
                self.assertIsNone(exc, out)
                self.assertEqual(len(net.run_posts), 2)

    def test_substring_and_prefix_tricks_are_rejected(self):
        for prov in ("deepinfra-free", "xdeepinfra", "deepinfra.nvidia", "deep", "infra", "deepinfra baseten",
                     "deepinfra,baseten", "nvidia/deepinfra", "baseten2", "DEEPINFRA\u200b", "deepinfra\x00",
                     "deepinfra/fp8", "free-baseten"):
            with self.subTest(prov=prov):
                ns, rec, exc, net, out = self.assert_refused(BASE + RUN, provider=prov, probes=2)
                self.assertFalse(ns["ok_to_run"])
                self.assertFalse(ns["PROBE_RESULT"]["ok"])


class TestNegatives(R6Base):
    def test_disallowed_provider(self):
        for prov in ("NVIDIA", "Venice", "openai", "Anthropic"):
            with self.subTest(prov=prov):
                ns, *_ = self.assert_refused(BASE + RUN, provider=prov, probes=2, contains="allowlist")
                self.assertFalse(ns["ok_to_run"])
                self.assertFalse(ns["PROBE_RESULT"]["ok"])

    def test_missing_provider_metadata_key(self):
        ns, *_ = self.assert_refused(BASE + RUN, provider=MISSING, probes=2, contains="no provider name")
        self.assertFalse(ns["ok_to_run"])
        self.assertIsNone(ns["PROBE_RESULT"]["provider"])

    def test_empty_and_blank_string(self):
        for prov in ("", "   ", "\n\t"):
            with self.subTest(prov=prov):
                self.assert_refused(BASE + RUN, provider=prov, probes=2, contains="no provider name")

    def test_non_string_provider(self):
        for prov in (None, 123, True, ["deepinfra"], {"name": "deepinfra"}, 1.5):
            with self.subTest(prov=prov):
                ns, *_ = self.assert_refused(BASE + RUN, provider=prov, probes=2, contains="no provider name")
                self.assertIsNone(ns["PROBE_RESULT"]["models"]["nemotron-3-ultra"]["provider"])

    def test_probe_raises(self):
        ns, *_ = self.assert_refused(BASE + RUN, raise_for=NEMO, probes=2)
        self.assertFalse(ns["ok_to_run"])

    def test_probe_http_error(self):
        ns, *_ = self.assert_refused(BASE + RUN, status=500, probes=2, contains="did not succeed")
        self.assertFalse(ns["ok_to_run"])

    def test_probe_call_itself_raises_out_of_call(self):
        # a budget refusal propagates out of call(); the probe must treat that as a failure without echoing its text
        ns, *_ = self.assert_refused(BASE + RUN, env={"RUN_BUDGET_USD": "0.0000001"}, probes=0, contains="raised")
        self.assertFalse(ns["ok_to_run"])

    def test_probe_never_run(self):
        # names deleted after the probe cell: as if it never ran in this session
        ns, *_ = self.assert_refused(BASE + ["del ok_to_run, PROBE_RESULT"] + RUN, probes=2, contains="not been run")

    def test_probe_cell_skipped_entirely(self):
        ns, rec, exc, net, out = execute([0, STUBS, 1, FAST, 2, 3, LIMIT, 5])
        self.assertIsInstance(exc, RuntimeError)
        self.assertTrue(str(exc).startswith("Refusing to run: "))
        self.assertEqual(rec.paid, [])
        self.assertEqual(net.posts, [])

    def test_hand_set_flag_without_a_probe_result(self):
        ns, rec, exc, net, out = execute([0, STUBS, 1, FAST, 2, 3, "ok_to_run = True", LIMIT, 5])
        self.assertIsInstance(exc, RuntimeError)
        self.assertTrue(str(exc).startswith("Refusing to run: "))
        self.assertEqual(rec.paid, [])

    def test_stale_flag_over_a_failed_probe(self):
        self.assert_refused(BASE + ["ok_to_run = True"] + RUN, provider="NVIDIA", probes=2)

    def test_model_mismatch(self):
        # a good probe, then the run model changes: the old ok_to_run must not carry over
        tamper = 'MODELS["nemotron-3-ultra"]["slug"] = "nvidia/some-other-model"'
        ns, *_ = self.assert_refused(BASE + [tamper] + RUN, probes=2, contains="current model")
        self.assertIs(ns["ok_to_run"], True)   # still the stale True; the run cell is what refuses

    def test_unrestricted_model_swapped_in(self):
        tamper = 'MODELS["opus-5-low"]["slug"] = "anthropic/claude-other"'
        self.assert_refused(BASE + [tamper] + RUN, probes=2, contains="current model")

    def test_extra_model_added_after_probe(self):
        tamper = 'MODELS["extra"] = dict(MODELS["opus-5-low"], slug="x/y")'
        self.assert_refused(BASE + [tamper] + RUN, probes=2, contains="does not cover")

    def test_allowlist_widened_after_probe(self):
        self.assert_refused(BASE + ['SAFE_PROVIDERS.append("otherhost")'] + RUN, probes=2, contains="allowlist changed")

    def test_provider_swapped_in_the_probe_result(self):
        tamper = 'PROBE_RESULT["models"]["nemotron-3-ultra"]["provider"] = "NVIDIA"'
        self.assert_refused(BASE + [tamper] + RUN, probes=2, contains="allowlisted provider")

    def test_provider_removed_from_the_probe_result(self):
        tamper = 'del PROBE_RESULT["models"]["nemotron-3-ultra"]["provider"]'
        self.assert_refused(BASE + [tamper] + RUN, probes=2, contains="allowlisted provider")

    def test_probe_result_not_a_dict(self):
        self.assert_refused(BASE + ["PROBE_RESULT = None"] + RUN, probes=2)


if __name__ == "__main__":
    unittest.main()
