"""Test configuration.

The project is laid out as a demo folder with most importable modules under
`nsck-demo/python/`. Some tests import modules as top-level (e.g. `import curiosity`)
while others use `from python import ...`.

This conftest ensures those imports work during pytest collection without
requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    root = Path(__file__).resolve().parent
    python_dir = root / "python"

    # Prepend so it wins over any similarly-named installed packages.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if python_dir.exists() and str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
