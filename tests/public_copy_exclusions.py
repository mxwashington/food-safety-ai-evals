"""The paths left out of the public copy, and the one rule for skipping a test case because of them. Standard library
only.

A test may skip a case only when that case's file is in EXCLUDED and is absent from this tree. Any other missing file
still fails, because the test opens it as before.
"""
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

EXCLUDED = frozenset({
    "benchmarks/probe-sonnet46-factcheck-leg.ipynb",
    "drb/drb-6-research-pro-runner.ipynb",
    "tests/test_public_copy_drb.py",
    "tests/test_public_copy_fsqa.py",
    "tests/test_public_copy_benchmarks.py",
})


def absent(rel):
    """True only when rel is an excluded path and the file is not in this tree."""
    return rel in EXCLUDED and not (REPO / rel).exists()


def present(paths):
    """paths without the excluded ones that are absent, order kept. In the private tree this is paths unchanged."""
    return [p for p in paths if not absent(p)]


def skip_if_absent(case, rel):
    """Skip the running test (or subtest) when rel is excluded and absent; otherwise do nothing."""
    if absent(rel):
        case.skipTest(f"{rel} is excluded from the public copy and absent here")
