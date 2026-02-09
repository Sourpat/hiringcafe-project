import os
import runpy
import sys
import time


def test_golden_regression_fast_demo():
    os.environ["OFFLINE_MODE"] = "1"
    os.environ["FAST_DEMO"] = "1"
    os.environ["FAST_DEMO_N"] = "20000"

    start = time.perf_counter()
    argv = [
        "scripts/check_golden.py",
        "--threshold",
        "2",
        "--single",
        "data science jobs",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path("scripts/check_golden.py", run_name="__main__")
    finally:
        sys.argv = old_argv

    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
