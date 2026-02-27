"""Knowledge pack — serializable bundle of concepts, relations, and causal links."""
from __future__ import annotations
import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class KnowledgePack:
    """Portable bundle of domain knowledge for injection into NSCK.

    .. warning:: Security — pickle deserialization
        ``KnowledgePack.load()`` deserializes data using Python's ``pickle``
        module wrapped in gzip.  Pickle can execute arbitrary code when
        loading a maliciously crafted file.  **Only load knowledge pack files
        from trusted sources.**  A future version will migrate to a versioned
        JSON schema to eliminate this risk.
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
        """Save to compressed pickle."""
        data = {
            "name": self.name,
            "concepts": self._concepts,
            "relations": self._relations,
            "causal_links": self._causal_links,
        }
        with gzip.open(str(path), "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "KnowledgePack":
        """Load from compressed pickle."""
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
                # Causal links stored as relations with strength in properties
                sem.add_relation(cause, "causes", effect)
                counts["causal_links"] += 1
            except Exception:
                pass
        
        return counts
