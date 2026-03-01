"""Pytest configuration for nsck_vision tests."""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_NSCK_VISION_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_NSCK_VISION_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")

for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)
