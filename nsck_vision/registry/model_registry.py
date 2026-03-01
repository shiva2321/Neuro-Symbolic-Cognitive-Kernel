"""ModelRegistry — persistent catalog of absorbed models."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Persistent JSON catalog of absorbed models.
    
    Tracks model metadata: domain, absorption stats, timestamps.
    """

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._path = registry_path
        self._records: Dict[str, Dict[str, Any]] = {}
        if registry_path and os.path.exists(registry_path):
            self.load(registry_path)

    def register(self, model_id: str, domain: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a model in the catalog."""
        self._records[model_id] = {
            "model_id": model_id,
            "domain": domain,
            "metadata": metadata or {},
            "registered_at": time.time(),
        }
        if self._path:
            self._auto_save()

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model record by ID."""
        return self._records.get(model_id)

    def list_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """List all models in a given domain."""
        return [r for r in self._records.values() if r["domain"] == domain]

    def all_models(self) -> List[Dict[str, Any]]:
        """Return all registered model records."""
        return list(self._records.values())

    def remove(self, model_id: str) -> None:
        """Remove a model from the registry."""
        self._records.pop(model_id, None)
        if self._path:
            self._auto_save()

    def save(self, path: str) -> None:
        """Save registry to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)

    def load(self, path: str) -> None:
        """Load registry from JSON file."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self._records = json.load(f)

    def _auto_save(self) -> None:
        if self._path:
            try:
                self.save(self._path)
            except Exception:
                pass
