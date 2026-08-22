"""Regression checks for selection-only hyperparameter grids."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MERGE_BASELINES = ROOT / "scripts" / "merge_baselines.py"


def _tree():
    return ast.parse(MERGE_BASELINES.read_text())


def _function(tree, name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _calls(function, name):
    return [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_grid_cells_cannot_call_full_evaluation():
    tree = _tree()
    assert not _calls(_function(tree, "sweep_arm"), "eval_merge")
    assert not _calls(_function(tree, "offer_baseline"), "eval_merge")


def test_full_grid_override_was_removed_and_selection_is_required():
    source = MERGE_BASELINES.read_text()
    legacy_flag = "eval_all" + "_cells"
    assert legacy_flag not in source
    n_select = next(
        call for call in ast.walk(_function(_tree(), "main"))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "--n_select"
    )
    required = next(
        keyword.value for keyword in n_select.keywords
        if keyword.arg == "required"
    )
    assert isinstance(required, ast.Constant) and required.value is True
