"""Cell selection is lexicographic: primary objective, then the tie-break.

On a small held-out fold the metric objective is quantized (accuracy on 16
examples moves in steps of 1/16), so distinct cells tie EXACTLY. Under the
metric rule the tie is broken by the normalised held-out loss of the same
predictions -- the finer reading of the same information -- and only then by
the smallest (eta, S). Without a secondary the rule must still be
deterministic. These tests pin that behaviour on hand-built objectives.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cv_protocol import select_cell


def test_primary_decides_when_not_tied():
    prim = {(1, 50): -0.70, (2, 50): -0.75, (4, 50): -0.72}
    sec = {(1, 50): 0.5, (2, 50): 0.9, (4, 50): 0.1}
    assert select_cell(prim, sec) == (2, 50), "a worse tie-break must not override the primary"


def test_secondary_breaks_exact_ties():
    prim = {(32, 20): -0.75, (32, 50): -0.75, (16, 50): -0.6719}
    sec = {(32, 20): 0.640, (32, 50): 0.548, (16, 50): 0.700}
    assert select_cell(prim, sec) == (32, 50)


def test_float_noise_does_not_hide_a_tie():
    """12/16 summed in two orders can differ in the last bits; that is not
    a real difference and must not pre-empt the tie-break."""
    a = sum([0.75, 0.6875, 0.8125, 0.75]) / 4
    b = sum([0.8125, 0.75, 0.75, 0.6875]) / 4
    prim = {(32, 20): -a, (32, 50): -b}
    sec = {(32, 20): 0.9, (32, 50): 0.1}
    assert select_cell(prim, sec) == (32, 50)


def test_no_secondary_is_deterministic_and_prefers_least_movement():
    prim = {(32, 50): -0.75, (32, 20): -0.75, (8, 50): -0.75}
    assert select_cell(prim, None) == (8, 50)
    assert select_cell(prim, {c: None for c in prim}) == (8, 50)


def test_loss_mode_unchanged():
    """With a strict ordering and no secondary, select_cell is plain argmin."""
    prim = {(0.5, 5): 0.9994, (4, 20): 0.9926, (32, 100): 1.31}
    assert select_cell(prim, None) == min(prim, key=prim.get)
