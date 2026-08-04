from __future__ import annotations

import ast
from pathlib import Path

from deconv.i18n import get_language, set_language, translate


def test_language_switch_and_dynamic_translation():
    set_language("en")
    assert get_language() == "en"
    assert translate("Load image") == "Load image"

    set_language("pl")
    assert get_language() == "pl"
    assert translate("Load image") == "Wczytaj obraz"
    dynamic = translate("Calculation input: 512×256 px; calculation PSF: 31×21 px, sum=1.")
    assert "Dane wejściowe do obliczeń" in dynamic
    assert "PSF obliczeniowa" in dynamic

    set_language("en")


def test_all_static_gui_sources_have_polish_display_text():
    set_language("pl")
    runtime_path = Path(__file__).parents[1] / "deconv" / "legacy_runtime.py"
    root = ast.parse(runtime_path.read_text(encoding="utf-8"))
    constructors = {"QPushButton", "QLabel", "QCheckBox", "QGroupBox", "ImageCanvas", "HistogramCanvas", "QAction"}
    methods = {"addRow", "addItem", "addItems", "setToolTip", "setPlaceholderText", "setWindowTitle"}
    technical = {
        "PSF", "TV", "Auto", "Epsilon", "DnCNN", "FFDNet", "DRUNet", "SCUNet", "Non-local Means",
        "center_of_mass", "geometric", "manual", "x", "y",
    }
    missing = []
    for node in ast.walk(root):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        values = []
        if name in constructors and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            values.append(node.args[0].value)
        elif name in methods:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    values.append(arg.value)
                elif isinstance(arg, (ast.List, ast.Tuple)):
                    values.extend(
                        item.value for item in arg.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    )
        for source in values:
            if source in technical or not any(ch.isalpha() for ch in source):
                continue
            if translate(source) == source:
                missing.append((node.lineno, source))
    set_language("en")
    assert not missing, missing


def test_dynamic_gui_templates_are_fully_translated():
    """Guard against mixed-language dynamic messages and substring corruption."""
    from deconv.i18n import _TEMPLATE_PL

    set_language("pl")
    forbidden = {
        " pending ", " apply ", " selected ", " calculation ", " stored ",
        " criteria ", " postprocessing ", " device=", " dtype=", " full array",
        " normalized ", " crop ", " preview ", " evaluations", " black=", " white=",
        " result ", " current ", " saved ", " failed ", " language:",
    }
    failures = []
    for source in _TEMPLATE_PL:
        sample = source.replace("{}", "X")
        translated = translate(sample)
        lowered = f" {translated.lower()} "
        residues = sorted(word for word in forbidden if word in lowered)
        if residues or "wycinekped" in translated or "sumaa" in translated:
            failures.append((source, translated, residues))
    set_language("en")
    assert not failures, failures


def test_polish_uses_disturbed_image_terminology():
    set_language("pl")
    assert translate("Degraded input") == "Obraz zaburzony"
    assert translate("Generate degraded input") == "Wygeneruj obraz zaburzony"
    assert "zdegradow" not in translate("Loaded image as measured/degraded input").lower()
    set_language("en")


def test_visible_runtime_strings_do_not_leave_english_sentence_fragments():
    runtime_path = Path(__file__).parents[1] / "deconv" / "legacy_runtime.py"
    root = ast.parse(runtime_path.read_text(encoding="utf-8"))
    constructors = {"QPushButton", "QLabel", "QCheckBox", "QGroupBox", "ImageCanvas", "HistogramCanvas", "QAction"}
    methods = {
        "addRow", "addItem", "addItems", "setToolTip", "setPlaceholderText", "setWindowTitle",
        "addTab", "setText", "showMessage", "append", "set_title", "set_xlabel", "set_ylabel",
        "warning", "information", "critical", "question", "getOpenFileName", "getSaveFileName",
        "getItem", "show_image", "show_histogram",
    }
    forbidden = (
        " pending ", " apply ", " selected ", " calculation ", " stored ",
        " criteria ", " postprocessing ", " device=", " dtype=", " full array",
        " normalized ", " crop ", " preview ", " evaluations", " black=", " white=",
        " result -", " current ", " saved ", " failed ", " language:",
    )

    def visible_sample(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(str(item.value) if isinstance(item, ast.Constant) else "X" for item in node.values)
        if isinstance(node, (ast.List, ast.Tuple)):
            values = []
            for item in node.elts:
                value = visible_sample(item)
                if isinstance(value, str):
                    values.append(value)
            return values
        return None

    set_language("pl")
    failures = []
    for node in ast.walk(root):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        values = []
        if name in constructors and node.args:
            value = visible_sample(node.args[0])
            if isinstance(value, str):
                values.append(value)
        if name in methods:
            start = 1 if name in {"warning", "information", "critical", "question", "getOpenFileName", "getSaveFileName", "getItem"} else 0
            for arg in node.args[start:]:
                value = visible_sample(arg)
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, list):
                    values.extend(value)
        for source in values:
            translated = translate(source)
            lowered = f" {translated.lower()} "
            residues = [term for term in forbidden if term in lowered]
            if residues or "wycinekped" in translated or "sumaa" in translated:
                failures.append((node.lineno, source, translated, residues))
    set_language("en")
    assert not failures, failures
