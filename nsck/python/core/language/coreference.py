"""Coreference Resolution module for NSCK V3."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import time
import python.core.vsa.hypervec_shim as hypervec_rs

HyperVector = hypervec_rs.HyperVector

PRONOUN_FEATURES: Dict[str, Dict[str, str]] = {
    "he":   {"gender": "male",   "animacy": "animate",   "number": "singular"},
    "him":  {"gender": "male",   "animacy": "animate",   "number": "singular"},
    "his":  {"gender": "male",   "animacy": "animate",   "number": "singular"},
    "she":  {"gender": "female", "animacy": "animate",   "number": "singular"},
    "her":  {"gender": "female", "animacy": "animate",   "number": "singular"},
    "hers": {"gender": "female", "animacy": "animate",   "number": "singular"},
    "it":   {"gender": "neuter", "animacy": "inanimate", "number": "singular"},
    "its":  {"gender": "neuter", "animacy": "inanimate", "number": "singular"},
    "they": {"gender": "any",    "animacy": "any",        "number": "plural"},
    "them": {"gender": "any",    "animacy": "any",        "number": "plural"},
    "their":{"gender": "any",    "animacy": "any",        "number": "plural"},
    "this": {"gender": "neuter", "animacy": "any",        "number": "singular"},
    "that": {"gender": "neuter", "animacy": "any",        "number": "singular"},
    "these":{"gender": "neuter", "animacy": "any",        "number": "plural"},
    "those":{"gender": "neuter", "animacy": "any",        "number": "plural"},
}

@dataclass
class EntityMention:
    name: str
    hv: HyperVector
    features: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


class EntityRegister:
    MAX_SIZE = 10

    def __init__(self):
        self._register: deque = deque(maxlen=self.MAX_SIZE)

    def register(self, name: str, hv, features: Dict[str, str]) -> None:
        mention = EntityMention(name=name, hv=hv, features=features)
        self._register.append(mention)

    def resolve(self, pronoun: str) -> Optional[EntityMention]:
        p = pronoun.lower()
        if p not in PRONOUN_FEATURES:
            return None
        pron_feats = PRONOUN_FEATURES[p]
        candidates = []
        for mention in reversed(list(self._register)):
            if _features_compatible(pron_feats, mention.features):
                candidates.append(mention)
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.timestamp)

    def get_all(self) -> List[EntityMention]:
        return list(self._register)

    def clear(self) -> None:
        self._register.clear()


def _features_compatible(pron_feats: Dict[str, str], entity_feats: Dict[str, str]) -> bool:
    for key, pron_val in pron_feats.items():
        if pron_val == "any":
            continue
        entity_val = entity_feats.get(key, "any")
        if entity_val == "any":
            continue
        if pron_val != entity_val:
            return False
    return True
