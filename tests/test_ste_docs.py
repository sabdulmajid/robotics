from __future__ import annotations

import importlib.util
from pathlib import Path


def test_ste_check_accepts_short_specific_text() -> None:
    module = load_module()
    failures = module.check_text(Path("status.md"), "# Status\n\nThe test used 150 episodes.\n")
    assert failures == []


def test_ste_check_rejects_long_and_subjective_text() -> None:
    module = load_module()
    text = "This very long sentence contains more than twenty five words because it repeats unnecessary details that do not help a reviewer understand the technical result or its limit."
    failures = module.check_text(Path("status.md"), text)
    assert any("maximum is 25" in failure for failure in failures)
    assert any("disallowed phrase 'very'" in failure for failure in failures)


def load_module():
    spec = importlib.util.spec_from_file_location("ste_check", "scripts/check_ste_docs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
