"""Data-path wording and request settings for benchmarks/bakeoff-r6-nemotron.ipynb. Offline: httpx is replaced by the
recorder from test_r6_probe, so no network and no paid call.

The provider restriction (provider.only, data_collection "deny", allow_fallbacks False) is sent on the Nemotron route
only. The Opus route carries a price limit and nothing else. These tests pin both requests and check that the
notebook's own wording says the same thing.
"""
import ast, io, json, pathlib, re, sys, tokenize, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_r6_probe as r6  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_PATH = ROOT / r6.NB
RESTRICTION_KEYS = ("only", "data_collection", "allow_fallbacks")
UNRESTRICTED_PHRASE = "The Opus route has no provider restriction"
WARNING_PHRASE = "Do not put\nconfidential or client material into prompts sent on the Opus route"
# A sentence that says every or all calls/requests send something, followed in the same text unit by one of the
# restriction settings. It is a false claim unless the subject is limited to the Nemotron route.
QUANTIFIED = re.compile(r"\b(every|all)\s+((?:[\w`.-]+\s+){0,3}?)(calls?|requests?)\b", re.I)
RESTRICTION_WORDS = re.compile(r"data_collection|allow_fallbacks|provider\.only|\brestrict", re.I)


def _requests():
    ns, rec, exc, net, out = r6.execute(r6.BASE + r6.RUN)
    if exc is not None:
        raise AssertionError(f"run did not complete: {exc!r}\n{out}")
    return net


def _text_units():
    """Every markdown cell, every contiguous block of # comments, and every docstring in the code cells."""
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    units = []
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            units.append((f"cell {i} markdown", src))
            continue
        if c["cell_type"] != "code":
            continue
        block, last = [], None
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                if block and last is not None and tok.start[0] != last + 1:
                    units.append((f"cell {i} comment", "\n".join(block)))
                    block = []
                block.append(tok.string.lstrip("#").strip())
                last = tok.start[0]
        if block:
            units.append((f"cell {i} comment", "\n".join(block)))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                doc = ast.get_docstring(node)
                if doc:
                    units.append((f"cell {i} docstring", doc))
    return units


class TestRequestSettings(unittest.TestCase):
    def test_nemotron_request_carries_the_provider_restriction(self):
        net = _requests()
        bodies = [b for b in net.posts if b[1]["model"] == r6.NEMO]
        self.assertEqual(len(bodies), 2)          # probe and run
        for _, b in bodies:
            prov = b["provider"]
            self.assertEqual(prov["only"], ["deepinfra", "baseten"])
            self.assertEqual(prov["data_collection"], "deny")
            self.assertIs(prov["allow_fallbacks"], False)

    def test_opus_request_carries_none_of_the_restriction_settings(self):
        net = _requests()
        bodies = [b for b in net.posts if b[1]["model"] == r6.OPUS]
        self.assertEqual(len(bodies), 2)          # probe and run
        for _, b in bodies:
            prov = b.get("provider") or {}
            for key in RESTRICTION_KEYS:
                self.assertNotIn(key, prov)
            self.assertEqual(set(prov), {"max_price"})


class TestDataPathWording(unittest.TestCase):
    def test_no_claim_that_every_or_all_calls_carry_the_restriction(self):
        bad = []
        for where, text in _text_units():
            for m in QUANTIFIED.finditer(text):
                if "nemotron" in m.group(2).lower():
                    continue
                window = text[m.start():m.start() + 400]
                if RESTRICTION_WORDS.search(window):
                    bad.append(f"{where}: {window[:120]!r}")
        self.assertEqual(bad, [], "claims every or all calls carry the provider restriction")

    def test_notebook_says_the_opus_route_is_unrestricted(self):
        nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
        intro = "".join(nb["cells"][0]["source"])
        self.assertIn(UNRESTRICTED_PHRASE, intro)
        self.assertIn(WARNING_PHRASE, intro)


if __name__ == "__main__":
    unittest.main()
