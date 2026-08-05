from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path


def test_pyproject_metadata_and_entry_points():
    root = Path(__file__).parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "dekonwolucje"
    assert project["license"] == "MIT"
    assert project["urls"]["Homepage"] == "https://github.com/rkotynski/dekonwolucje"
    assert project["scripts"]["dekonwolucje"] == "deconv.main:main"
    assert project["gui-scripts"]["dekonwolucje-gui"] == "deconv.main:main"
    assert (root / "LICENSE").exists()
    assert (root / "docs" / "USER_GUIDE_EN.md").exists()
    assert (root / "docs" / "USER_GUIDE_PL.md").exists()
    assert (root / "docs" / "screenshots" / "README.md").exists()
    assert (root / "docs" / "API_EN.md").exists()
    assert (root / "docs" / "API_PL.md").exists()
    assert (root / "examples" / "wiener_motion_blur.py").exists()


def test_english_pdf_documentation_is_packaged():
    root = Path(__file__).parents[1]
    pdf = root / "docs" / "Deconvolution_GUI_and_Methods_EN.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 100_000


def test_pdf_source_has_no_named_author_and_contains_ai_disclosure():
    root = Path(__file__).parents[1]
    tex = (root / "docs" / "Deconvolution_GUI_and_Methods_EN.tex").read_text(encoding="utf-8")
    assert "Rafał Kotyński" not in tex
    assert "University of Warsaw, Faculty of Physics" not in tex
    assert "pdfauthor={}" in tex
    assert "AI-assisted development disclosure" in tex
    assert "large language models (LLMs)" in tex
    assert "Relation to the classical Kaczmarz method" in tex
    assert "03-kaczmarz-settings.png" in tex
    assert "Python API" in tex
    assert "wiener_motion_blur.py" in tex
