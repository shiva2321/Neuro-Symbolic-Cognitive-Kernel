"""Knowledge pack — serializable bundle of concepts, relations, and causal links."""
from __future__ import annotations
import base64
import gzip
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 2


def _hv_to_json(hv: Any) -> Optional[str]:
    """Serialize a HyperVector to a base64 string (safe, no pickle)."""
    if hv is None:
        return None
    try:
        bits = np.asarray(hv.bits, dtype=np.int8)
        return base64.b64encode(bits.tobytes()).decode("ascii")
    except Exception:
        return None


def _json_to_hv(s: Optional[str]) -> Any:
    """Deserialize a HyperVector from a base64 string."""
    if s is None:
        return None
    try:
        import python.core.vsa.hypervec_shim as hv_mod
        bits = np.frombuffer(base64.b64decode(s), dtype=np.int8)
        return hv_mod.HyperVector.from_bits(bits)
    except Exception:
        return None


class KnowledgePack:
    """Portable bundle of domain knowledge for injection into NSCK.

    Serialization uses gzip+JSON (schema_version=2) for safe deserialization.
    Legacy gzip+pickle files (V14/V15) are still loaded with a
    ``DeprecationWarning`` for backward compatibility.
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self._concepts: List[Tuple[str, Dict, Any]] = []  # (name, properties, hv)
        self._relations: List[Tuple[str, str, str]] = []  # (src, rel, dst)
        self._causal_links: List[Tuple[str, str, float]] = []  # (cause, effect, strength)

    def add_concept(self, name: str, properties: Dict, hv: Any = None) -> None:
        self._concepts.append((name, dict(properties), hv))

    def add_relation(self, src: str, rel: str, dst: str) -> None:
        self._relations.append((src, rel, dst))

    def add_causal_link(self, cause: str, effect: str, strength: float = 1.0) -> None:
        self._causal_links.append((cause, effect, float(strength)))

    def save(self, path: str) -> None:
        """Save to gzip+JSON (schema_version=2, safe serialization)."""
        concepts_json = [
            [name, props, _hv_to_json(hv)]
            for name, props, hv in self._concepts
        ]
        data = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "concepts": concepts_json,
            "relations": [list(r) for r in self._relations],
            "causal_links": [list(c) for c in self._causal_links],
        }
        with gzip.open(str(path), "wt", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "KnowledgePack":
        """Load from gzip+JSON (schema_version=2) or legacy gzip+pickle.

        Legacy pickle files are loaded with a ``DeprecationWarning``.
        """
        # Try JSON first
        try:
            with gzip.open(str(path), "rt", encoding="utf-8") as f:
                data = json.load(f)
            pack = cls(name=data.get("name", "unnamed"))
            raw_concepts = data.get("concepts", [])
            pack._concepts = [
                (c[0], c[1], _json_to_hv(c[2] if len(c) > 2 else None))
                for c in raw_concepts
            ]
            pack._relations = [tuple(r) for r in data.get("relations", [])]
            pack._causal_links = [tuple(c) for c in data.get("causal_links", [])]
            return pack
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            pass  # Fall through to pickle fallback

        # Fallback: legacy pickle (V14/V15 backward compatibility)
        warnings.warn(
            f"Loading KnowledgePack from pickle ({path}). "
            "Pickle deserialization can execute arbitrary code. "
            "Re-save this file with KnowledgePack.save() to upgrade to JSON (schema_version=2).",
            DeprecationWarning,
            stacklevel=2,
        )
        import pickle
        with gzip.open(str(path), "rb") as f:
            data = pickle.load(f)
        pack = cls(name=data.get("name", "unnamed"))
        pack._concepts = data.get("concepts", [])
        pack._relations = data.get("relations", [])
        pack._causal_links = data.get("causal_links", [])
        return pack

    def inject_into(self, engine: Any) -> Dict[str, int]:
        """Inject concepts and relations into a CognitiveEngine. Returns counts."""
        counts = {"concepts": 0, "relations": 0, "causal_links": 0}

        # Get semantic memory
        sem = getattr(engine, "semantic_memory", None)
        if sem is None:
            return counts

        for name, props, hv in self._concepts:
            try:
                sem.add_concept(name, props)
                if hv is not None:
                    sem.concept_hvs[name] = hv
                counts["concepts"] += 1
            except Exception:
                pass

        for src, rel, dst in self._relations:
            try:
                sem.add_relation(src, rel, dst)
                counts["relations"] += 1
            except Exception:
                pass

        for cause, effect, strength in self._causal_links:
            try:
                sem.add_relation(cause, "causes", effect)
                counts["causal_links"] += 1
            except Exception:
                pass

        return counts
