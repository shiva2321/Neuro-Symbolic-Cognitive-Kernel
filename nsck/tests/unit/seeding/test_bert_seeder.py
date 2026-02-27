"""Tests for BertSeeder (V16 Initiative 4)."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from python.core.seeding.bert_seeder import BertSeeder


def test_bert_seeder_raises_without_transformers(monkeypatch):
    """BertSeeder.seed() must raise RuntimeError if transformers not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    seeder = BertSeeder()
    with pytest.raises(RuntimeError, match="transformers"):
        seeder.seed(object())  # substrate doesn't matter — will fail early
