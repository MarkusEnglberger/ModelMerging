"""Parallel MBPP scoring must agree exactly with serial scoring.

Uses real MBPP+ rows and a mix of correct / wrong / crashing / hanging
candidate programs, so the timeout and exception paths are covered too.
Run: python tests/test_mbpp_parallel_scoring.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.causal_lm import _mbpp_score, extract_code, run_python_tests  # noqa: E402


def serial_score(rows, texts):
    hits = sum(int(run_python_tests(extract_code(t), [r["test"]], timeout=30.0))
               for r, t in zip(rows, texts))
    return hits / max(len(rows), 1)


def main():
    from datasets import load_dataset
    ds = load_dataset("evalplus/mbppplus", split="test")  # resolved via $HF_HOME
    rows = list(ds.select(range(40)))

    # candidate generations: the reference solution for even indices (should
    # pass), a deliberately wrong body for odd ones, plus explicit crash and
    # hang cases to exercise the exception / timeout branches.
    texts = []
    for i, r in enumerate(rows):
        if i % 2 == 0:
            texts.append("```python\n" + r["code"] + "\n```")
        else:
            texts.append("```python\ndef wrong(*a, **k):\n    return None\n```")
    texts[3] = "```python\nraise RuntimeError('boom')\n```"
    texts[5] = "```python\nimport time\ntime.sleep(45)\n```"   # exceeds timeout
    texts[7] = "```python\nthis is not python\n```"

    t0 = time.time()
    ser = serial_score(rows, texts)
    t_ser = time.time() - t0

    t0 = time.time()
    par = _mbpp_score(rows, texts)["primary"]
    t_par = time.time() - t0

    print(f"serial   {ser:.6f}  ({t_ser:.1f}s)")
    print(f"parallel {par:.6f}  ({t_par:.1f}s)  speedup x{t_ser / max(t_par, 1e-9):.1f}")
    assert par == ser, f"score mismatch: parallel {par} != serial {ser}"

    # empty input must not divide by zero
    assert _mbpp_score([], [])["primary"] == 0.0
    print("OK: parallel scoring matches serial exactly")


if __name__ == "__main__":
    main()
