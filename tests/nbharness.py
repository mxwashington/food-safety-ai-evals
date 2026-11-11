"""Shared helper for the notebook tests. Standard library only. No network, no paid calls.

run_cells() executes chosen code cells of a notebook, in order, in one namespace, with every outbound path replaced
by a recorder: subprocess.run/Popen/check_call/check_output, os.system, requests/httpx/urllib, openai, boto3, modal
and anthropic stubs, and sockets. Tests then assert on rec.paid (anything that would have left the machine or started
a paid process) and on the exception the cells raised.
"""
import builtins, contextlib, io, json, os, pathlib, socket, subprocess, sys, tempfile, types
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parent.parent


def load_cells(nb_path):
    nb = json.load(open(REPO / nb_path, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"], nb


class Recorder:
    def __init__(self):
        self.paid = []        # calls that would be a paid request or start a paid process
        self.subprocess = []  # every subprocess invocation (argv list or string)
        self.other = []

    def note_paid(self, what):
        self.paid.append(what)


class _Net(types.ModuleType):
    def __init__(self, name, rec):
        super().__init__(name)
        self.__path__ = []
        self._rec = rec

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        rec, name = self._rec, self.__name__

        def call(*a, **k):
            rec.note_paid(f"{name}.{attr}")
            return mock.MagicMock(name=f"{name}.{attr}")
        call.__name__ = attr
        return call


def run_cells(nb_path, cell_ids, env=None, cwd=None, pre_exec=None, post_hooks=None):
    """Run the notebook's code cells with the given indexes (indexes among CODE cells, in the order given).
    Returns (namespace, recorder, exception_or_None). A cell that raises ends the run; the exception is returned."""
    cells, _ = load_cells(nb_path)
    rec = Recorder()
    ns = {"__name__": "__main__"}
    mods = {}
    for n in ["openai", "anthropic", "boto3", "botocore", "botocore.config", "botocore.exceptions", "modal",
              "requests", "httpx", "aiohttp"]:
        mods[n] = _Net(n, rec)
    mods["openai"].OpenAI = lambda *a, **k: _Net("openai.client", rec)
    mods["botocore.config"].Config = lambda *a, **k: mock.MagicMock()
    mods["boto3"].client = lambda *a, **k: _Net("boto3.client", rec)
    tmp = tempfile.mkdtemp(prefix="nbtest_")
    cwd = cwd or tmp
    base_env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "SPEND_LEDGER_DIR": os.path.join(tmp, "ledger")}
    full_env = dict(base_env, **(env or {}))

    def fake_run(cmd, *a, **k):
        rec.subprocess.append(cmd)
        argv = cmd if isinstance(cmd, (list, tuple)) else str(cmd).split()
        prog = os.path.basename(str(argv[0])) if argv else ""
        # git reads are allowed through to the real thing only when a test sets rec.allow_git
        if prog == "git" and getattr(rec, "allow_git", False):
            return real_run(cmd, *a, **k)
        # A test may answer the read-only "git ... rev-parse HEAD" by setting rec.git_head (in pre_exec). It is a
        # local read, so it is recorded in rec.subprocess and not in rec.paid.
        if prog == "git" and getattr(rec, "git_head", None) is not None and list(argv[-2:]) == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=rec.git_head + "\n", stderr="")
        rec.note_paid("subprocess:" + " ".join(map(str, argv))[:200])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    real_run = subprocess.run

    def no_sock(*a, **k):
        rec.note_paid("socket")
        raise OSError("sockets are blocked in tests")

    cm = contextlib.ExitStack()
    cm.enter_context(mock.patch.dict(sys.modules, mods))
    cm.enter_context(mock.patch.dict(os.environ, full_env, clear=True))
    cm.enter_context(mock.patch.object(subprocess, "run", fake_run))
    cm.enter_context(mock.patch.object(subprocess, "Popen", lambda *a, **k: (rec.note_paid("Popen"), mock.MagicMock())[1]))
    cm.enter_context(mock.patch.object(subprocess, "check_call", lambda *a, **k: rec.note_paid("check_call")))
    cm.enter_context(mock.patch.object(subprocess, "check_output", lambda *a, **k: (rec.note_paid("check_output"), b"")[1]))
    cm.enter_context(mock.patch.object(os, "system", lambda c: (rec.note_paid("os.system:" + str(c)[:100]), 0)[1]))
    cm.enter_context(mock.patch.object(socket, "socket", no_sock))
    cm.enter_context(mock.patch.object(socket, "create_connection", no_sock))
    old = os.getcwd()
    exc = None
    try:
        os.chdir(cwd)
        if pre_exec:
            pre_exec(ns, rec)
        with cm:
            for i in cell_ids:
                src = "\n".join(l for l in cells[i].splitlines() if not l.startswith(("!", "%")))
                try:
                    exec(compile(src, f"{nb_path}[cell {i}]", "exec"), ns)
                except BaseException as e:  # SystemExit included
                    exc = e
                    break
            if post_hooks:
                post_hooks(ns, rec)
    finally:
        os.chdir(old)
    return ns, rec, exc
