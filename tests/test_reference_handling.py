from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from deconv.core.runtime import GrayImage, reference_metrics_available


def test_reference_metrics_require_an_independent_reference() -> None:
    disturbed = GrayImage(
        np.full((8, 8), 0.25, dtype=np.float64),
        name="measured",
        metadata={"measured_input": True, "_preserve_intensity": True},
    )
    assert not reference_metrics_available({"degraded": disturbed, "reference_available": False})

    reference = GrayImage(
        np.full((8, 8), 0.75, dtype=np.float64),
        name="reference",
        metadata={"reference_input": True, "_preserve_intensity": True},
    )
    assert reference_metrics_available(
        {"image": reference, "degraded": disturbed, "reference_available": True}
    )

    identical_reference = GrayImage(
        disturbed.data.copy(),
        name="not-independent",
        metadata={"reference_input": True, "_preserve_intensity": True},
    )
    assert not reference_metrics_available(
        {"image": identical_reference, "degraded": disturbed, "reference_available": True}
    )


def test_gui_load_actions_keep_disturbed_and_reference_roles_separate() -> None:
    runtime_path = Path(__file__).parents[1] / "deconv" / "legacy_runtime.py"
    root = ast.parse(runtime_path.read_text(encoding="utf-8"))
    functions = {
        node.name: ast.get_source_segment(runtime_path.read_text(encoding="utf-8"), node) or ""
        for node in root.body
        if isinstance(node, ast.ClassDef) and node.name == "LoadGenerateTab"
        for node in node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    disturbed_source = functions["load_disturbed_image"]
    reference_source = functions["load_reference_image"]

    assert 'self.state.pop("image", None)' in disturbed_source
    assert 'self.state["reference_available"] = False' in disturbed_source
    assert 'self.state["degraded"]' in disturbed_source

    assert 'self.state["image"]' in reference_source
    assert 'self.state["reference_available"] = True' in reference_source
    assert 'self.state.pop("degraded", None)' not in reference_source
