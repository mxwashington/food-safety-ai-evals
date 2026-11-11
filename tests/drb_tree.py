"""Helpers for tests/test_drb_grading.py. Standard library only. No network, no paid calls.

- exec_cell(): runs a notebook cell statement by statement, so a test can set a module constant right after the
  statement that defines it ("exec definitions, set constant, run the entry").
- run(): runs chosen code cells of a notebook inside nbharness.run_cells' recording stubs, continuing past a refusal so
  a later cell can be tried in the same namespace.
- build_repo(): a tiny fake grader checkout made with git init in a temporary directory. Nothing is cloned.
"""
import ast, json, os, pathlib, subprocess, tempfile, time, types
from unittest import mock

import nbharness

_REAL_POPEN = subprocess.Popen   # captured before any stub: nbharness replaces Popen, which also breaks a real git run

GRADING_NBS = {
    "deep": "drb/drb-deep-investigation-grading.ipynb",
    "quick": "drb/drb-quick-scan-grading.ipynb",
    "part2": "drb/drb-deep-investigation-part-2.ipynb",
}
# For each notebook: the code cell that holds the grader checkout code, the name of its verify function, and the name
# of its constant for the grader repository URL.
GRADER_CELL = {"deep": (0, "verify_grader_checkout", "GRADER_REPO_URL"),
               "quick": (0, "ensure_grader_checkout", "GRADER_REPO_URL"),
               "part2": (4, "verify_grader_checkout", "GRADER_REPO")}
FORBIDDEN_NAMES = ("SPEND_LEDGER_DIR", "RUN_BUDGET_USD", "JOB_WORST_CASE_USD", "runner.lock")
KEYS = {"OPENROUTER_API_KEY": "test-key", "JINA_API_KEY": "test-key", "BENCHMARK_API_KEY": "test-key"}


def _defined(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    return []


def _is_definition(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Assign):
        try:
            ast.literal_eval(node.value)
            return True
        except ValueError:
            return False
    return False


def exec_cell(src, ns, label, patch=None, defs_only=False):
    """Exec src one top-level statement at a time. After a statement that defines a name in patch, that name is set
    to the patch value. With defs_only, only imports, functions, classes and constants are executed."""
    src = "\n".join(l for l in src.splitlines() if not l.startswith(("!", "%")))
    for node in ast.parse(src).body:
        if defs_only and not _is_definition(node):
            continue
        exec(compile(ast.Module([node], []), label, "exec"), ns)
        for n in _defined(node):
            if patch and n in patch:
                ns[n] = patch[n]


def cell_source(nb_path, idx):
    cells, _ = nbharness.load_cells(nb_path)
    return cells[idx]


def run(nb_key, steps, env=None, cwd=None, patch=None, setup=None, allow_git=False, defs_only=False):
    """Run steps in order inside the stubs. A step is a code-cell index or a callable fn(ns, rec). A step that raises
    is recorded and the next step still runs. Returns a namespace with ns, rec, excs (one per cell step)."""
    nb_path = GRADING_NBS.get(nb_key, nb_key)
    cells, _ = nbharness.load_cells(nb_path)
    out = types.SimpleNamespace(ns=None, rec=None, excs=[], cwd=cwd)

    def pre(ns, rec):
        rec.allow_git = allow_git
        if setup:
            setup(ns, rec)

    def hook(ns, rec):
        out.ns, out.rec = ns, rec
        harness_popen = subprocess.Popen

        def popen(*a, **k):
            argv = a[0] if a else k.get("args")
            if getattr(rec, "allow_git", False) and isinstance(argv, (list, tuple)) and argv \
                    and os.path.basename(str(argv[0])) == "git":
                return _REAL_POPEN(*a, **k)       # only git reads are real, and only when the test allows them
            return harness_popen(*a, **k)

        popen_patch = mock.patch.object(subprocess, "Popen", popen)
        popen_patch.start()
        try:
            run_steps(ns, rec)
        finally:
            popen_patch.stop()

    def run_steps(ns, rec):
        for s in steps:
            if callable(s):
                s(ns, rec)
                continue
            try:
                exec_cell(cells[s], ns, f"{nb_path}[cell {s}]", patch, defs_only)
                out.excs.append(None)
            except BaseException as e:
                out.excs.append(e)

    with mock.patch.object(time, "sleep", lambda s: None):
        nbharness.run_cells(nb_path, [], env=env, cwd=cwd, pre_exec=pre, post_hooks=hook)
    return out


def ledger_lines(ledger_dir):
    lines = []
    for p in pathlib.Path(ledger_dir).rglob("spend_ledger.jsonl"):
        lines += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return lines


def reservations(ledger_dir):
    return [l for l in ledger_lines(ledger_dir) if l.get("state") == "reserved"]


def _git_env(home):
    return {"PATH": os.environ.get("PATH", ""), "HOME": home, "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def git(repo, *args, home=None):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True,
                          env=_git_env(home or repo)).stdout


def build_repo(parent, origin_url):
    """parent/deep_research_bench: a committed tiny fake grader (code, requirements, data dir, one executable tool,
    a .gitignore). Returns (repo_path, head_sha). The remote origin is origin_url; nothing is fetched."""
    d = os.path.join(parent, "deep_research_bench")
    os.makedirs(d)
    files = {
        "requirements.txt": "requests\n",
        "deepresearch_bench_race.py": "print('grade')\n",
        "utils/__init__.py": "",
        "utils/api.py": "KEY = None\n",
        "data/test_data/raw_data/seed.jsonl": '{"id": 1}\n',
        "data/prompt_data/query.jsonl": '{"id": 1, "prompt": "q"}\n',
        "tool.sh": "#!/bin/sh\necho tool\n",
        ".gitignore": "local_*.py\n",
    }
    for rel, text in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
    os.chmod(os.path.join(d, "tool.sh"), 0o755)
    subprocess.run(["git", "init", "-q", d], check=True, capture_output=True, env=_git_env(parent))
    git(d, "remote", "add", "origin", origin_url, home=parent)
    git(d, "add", "-A", home=parent)
    git(d, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init", home=parent)
    return d, git(d, "rev-parse", "HEAD", home=parent).strip()


def write(repo, rel, text="x\n", mode=None):
    p = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)
    if mode is not None:
        os.chmod(p, mode)
    return p


def tmpdir():
    return tempfile.mkdtemp(prefix="drbtree_")
