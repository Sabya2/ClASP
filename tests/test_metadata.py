from pathlib import Path

from srcs.version import __version__


def test_version_is_consistent_across_release_metadata():
    root = Path(__file__).resolve().parents[1]
    assert f"version: {__version__}" in (root / "CITATION.cff").read_text()
    assert __version__ in (root / "CHANGELOG.md").read_text()
