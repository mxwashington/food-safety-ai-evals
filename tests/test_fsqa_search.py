"""The web-search paths in phase1-eval-gen-v2, phase1-eval-gen-v3 and phase5-testset-rebuild are refused
until search cost is bounded. Mocked only: requests is replaced by a recorder, no network, no paid call.

Each test executes the notebook's real code cells (the config cell, the spend guard cell and the request-building
cell) through tests/nbharness.py, with the cell tails that would run a stage cut off. nbharness.py is not edited.
"""
import json, pathlib, sys, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nbharness  # noqa: E402

REAL_LOAD = nbharness.load_cells
SEED_V = {"batch_id": "b1", "section": "117.1", "n_questions": 2, "archetype": "x", "domain": "d"}
SEED_5 = {"batch_id": "b5", "section": "117.1", "domain": "d", "archetype": "x", "ask": 2, "label": "L", "text": "t"}
OK_CONTENT = json.dumps({"questions": []})
BANNED = ("SPEND_LEDGER_DIR", "RUN_BUDGET_USD", "JOB_WORST_CASE_USD", "runner.lock")

CFG = {
    "v2": dict(nb="fsqa-pipeline/phase1-eval-gen-v2.ipynb", cells=[1, 2, 5, 6, 7], cut={6: '\n_msg = ['}),
    "v3": dict(nb="fsqa-pipeline/phase1-eval-gen-v3.ipynb", cells=[0, 1, 4, 5, 6], cut={5: '\n_msg = ['}),
    "p5": dict(nb="fsqa-pipeline/phase5-testset-rebuild.ipynb", cells=[0, 1, 5, 9], cut={9: '\ntodo = [s for s'}),
}


class FakeResp:
    status_code = 200
    headers = {}
    text = "{}"

    def json(self):
        return {"choices": [{"message": {"content": OK_CONTENT}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001}, "provider": "amazon-bedrock"}


class Gen:
    """Rows are stored the way the notebook's checkpoint would hold them; nothing is written to disk."""
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def has(self, k):
        return False


def run(key, after, set_vars=None):
    """Execute the notebook's cells, then call `after(ns, rec, sent)` with requests.post stubbed. Returns (ns, rec, sent, exc)."""
    c = CFG[key]
    cells, nb = REAL_LOAD(c["nb"])
    cells = list(cells)
    for i, marker in c["cut"].items():
        assert marker in cells[i], (key, i)
        cells[i] = cells[i].split(marker)[0]
        if key == "p5":   # the config cell defines the real section list; the stub replaces it for this one cell
            cells[i] = "NEW_SECTIONS = _STUB_SECTIONS\n" + cells[i]
    out = {}
    sent = []

    def pre(ns, rec):
        import os
        os.makedirs("local-data", exist_ok=True)
        ns["CFR_TEXT"] = {"117.1": {"heading": "h", "text": "section text", "chars": 12}}
        ns["CFR"] = {"117.1": {"chunks": [("L", "t")]}}
        ns["_STUB_SECTIONS"] = {"Dom": ["117.1"]}
        ns["ARCHETYPES"] = ["x"]
        ns["q_count"] = lambda n: 2
        ns["gen_ckpt"] = Gen()

    def post(ns, rec):
        def fake_post(url, headers=None, json=None, timeout=None, **k):
            sent.append(json)
            rec.note_paid("requests.post")
            return FakeResp()
        ns["requests"].post = fake_post
        for k, v in (set_vars or {}).items():
            ns[k] = v
        try:
            out["result"] = after(ns, rec, sent)
        except BaseException as e:
            out["exc"] = e
    with mock.patch.object(nbharness, "load_cells", lambda p: (cells, nb)), \
            mock.patch("time.sleep", lambda s: None):
        ns, rec, exc = nbharness.run_cells(c["nb"], c["cells"], env={"OPENROUTER_API_KEY": "test-key"},
                                           pre_exec=pre, post_hooks=post)
    assert exc is None, repr(exc)
    return ns, rec, sent, out


def generate(key):
    if key == "p5":
        return lambda ns, rec, sent: ns["generate_batch"](SEED_5)
    return lambda ns, rec, sent: ns["generate_batch"](SEED_V)


class Base:
    key = None

    def check_refused(self, ns, rec, sent, exc):
        self.assertIsInstance(exc, RuntimeError)
        self.assertTrue(str(exc).startswith("Refusing to run:"), str(exc))
        for w in ("billed separately", "more than once", "result tokens", "does not bound", "stay"):
            self.assertIn(w, str(exc))
        for b in BANNED:
            self.assertNotIn(b, str(exc))
        self.assertEqual(rec.paid, [])
        self.assertEqual(sent, [])
        self.assertEqual(ns["_ledger"], {})                    # nothing reserved
        self.assertFalse(__import__("os").path.exists(ns["LEDGER_PATH"]) and open(ns["LEDGER_PATH"]).read().strip())

    def test_shipped_constant_is_false(self):
        ns, rec, sent, out = run(self.key, lambda ns, rec, sent: ns["SEARCH_COST_BOUNDED"])
        self.assertIs(out["result"], False)
        self.assertTrue(ns["USE_WEB_SEARCH"])                   # the option is still in the code

    def test_search_request_refused_before_reserve(self):
        ns, rec, sent, out = run(self.key, generate(self.key))
        self.check_refused(ns, rec, sent, out.get("exc"))

    def test_guarded_post_refuses_every_search_shape(self):
        shapes = [{"tools": [{"type": "openrouter:web_search"}]},
                  {"tools": [{"type": "web_search_preview"}]},
                  {"plugins": [{"id": "web"}]},
                  {"web_search_options": {}},
                  {"plugins": [{"id": "web-search", "max_results": 3}]},
                  {"tools": [{"type": "function", "function": {"name": "web_lookup"}}]},
                  {"web_search": True},
                  {"model": "anthropic/claude-opus-5:online:nitro"}]

        def after(ns, rec, sent):
            res = []
            for extra in shapes + [{"model": "anthropic/claude-opus-5:online"}]:
                body = {"model": "anthropic/claude-opus-5", "max_tokens": 10,
                        "messages": [{"role": "user", "content": "hi"}], **extra}
                try:
                    ns["guarded_post"](body, {}, purpose="probe")
                    res.append(None)
                except BaseException as e:
                    res.append(e)
            return res
        ns, rec, sent, out = run(self.key, after)
        self.assertEqual(len(out["result"]), 9)
        for e in out["result"]:
            self.check_refused(ns, rec, sent, e)

    def test_non_search_request_unchanged(self):
        ns, rec, sent, out = run(self.key, generate(self.key), {"USE_WEB_SEARCH": False})
        self.assertNotIn("exc", out, out.get("exc"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(rec.paid, ["requests.post"])
        self.assertNotIn("tools", sent[0])
        self.assertNotIn("plugins", sent[0])
        self.assertEqual(len(ns["_ledger"]), 1)                 # reserved and settled as before
        self.assertEqual(next(iter(ns["_ledger"].values()))["state"], "settled")
        self.assertIn("max_price", sent[0]["provider"])

    def test_guard_is_the_only_block(self):
        ns, rec, sent, out = run(self.key, generate(self.key), {"SEARCH_COST_BOUNDED": True})
        self.assertNotIn("exc", out, out.get("exc"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(rec.paid, ["requests.post"])
        self.assertTrue(sent[0].get("tools") or sent[0].get("plugins"))
        self.assertEqual(len(ns["_ledger"]), 1)

    def test_truthy_non_true_value_does_not_unlock(self):
        ns, rec, sent, out = run(self.key, generate(self.key), {"SEARCH_COST_BOUNDED": 1})
        self.check_refused(ns, rec, sent, out.get("exc"))


class V2(Base, unittest.TestCase):
    key = "v2"

    def test_build_payload_refuses(self):
        ns, rec, sent, out = run("v2", lambda ns, rec, sent: ns["build_payload"](SEED_V))
        self.check_refused(ns, rec, sent, out.get("exc"))

    def test_probe_call_site_refuses(self):
        def after(ns, rec, sent):
            base = {"model": ns["MODEL_SLUG"], "messages": [{"role": "user", "content": "OK"}], "max_tokens": 50,
                    "provider": ns["PROVIDER_CONFIG"]}
            return ns["probe"]("+ web search", {**base, "tools": ns["WEB_SEARCH_TOOL"]})
        ns, rec, sent, out = run("v2", after)
        self.check_refused(ns, rec, sent, out.get("exc"))


class V3(V2):
    key = "v3"

    def test_build_payload_refuses(self):
        ns, rec, sent, out = run("v3", lambda ns, rec, sent: ns["build_payload"](SEED_V))
        self.check_refused(ns, rec, sent, out.get("exc"))

    def test_probe_call_site_refuses(self):
        def after(ns, rec, sent):
            base = {"model": ns["MODEL_SLUG"], "messages": [{"role": "user", "content": "OK"}], "max_tokens": 50,
                    "provider": ns["PROVIDER_CONFIG"]}
            return ns["probe"]("+ both", {**base, "tools": ns["WEB_SEARCH_TOOL"]})
        ns, rec, sent, out = run("v3", after)
        self.check_refused(ns, rec, sent, out.get("exc"))


class Phase5(Base, unittest.TestCase):
    key = "p5"

    def test_call_or_refuses_plugin_body(self):
        def after(ns, rec, sent):
            body = {"model": ns["GENERATOR"]["model"], "provider": ns["GEN_PROVIDER"], "max_tokens": 10,
                    "plugins": [{"id": "web"}], "messages": [{"role": "user", "content": "hi"}]}
            return ns["call_or"](body)          # must not be swallowed into the retry loop
        ns, rec, sent, out = run("p5", after)
        self.check_refused(ns, rec, sent, out.get("exc"))

    def test_call_or_without_plugin_still_sends(self):
        def after(ns, rec, sent):
            body = {"model": ns["GENERATOR"]["model"], "provider": ns["GEN_PROVIDER"], "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]}
            return ns["call_or"](body)
        ns, rec, sent, out = run("p5", after)
        self.assertTrue(out["result"]["ok"])
        self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    unittest.main()
