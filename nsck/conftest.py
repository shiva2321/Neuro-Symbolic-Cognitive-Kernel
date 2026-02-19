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
    
    # Only add root to sys.path for package-qualified imports like python.core.vsa
    # Adding python/ would cause conflicts since "import python.x" would try to find python/python/x
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
