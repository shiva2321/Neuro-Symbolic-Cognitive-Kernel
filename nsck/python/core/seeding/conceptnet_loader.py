"""ConceptNet CSV → KnowledgePack loader."""
from __future__ import annotations
import csv
import io
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("nsck.seeding.conceptnet_loader")

# Relation mapping: ConceptNet relation → NSCK relation
RELATION_MAP: Dict[str, str] = {
    "IsA": "is_a",
    "HasProperty": "has_property",
    "CapableOf": "capable_of",
    "UsedFor": "used_for",
    "Causes": "causes",
    "HasPart": "has_part",
    "PartOf": "part_of",
    "AtLocation": "at_location",
}

_DEFAULT_RELATIONS = list(RELATION_MAP.keys())


def _en_concept(uri: str) -> Optional[str]:
    """Extract English concept name from a ConceptNet URI, or None if not English."""
    if not uri.startswith("/c/en/"):
        return None
    parts = uri.split("/")
    # /c/en/<name>[/<pos>[/<sense>]]
    if len(parts) < 4:
        return None
    return parts[3].replace("_", " ")


class ConceptNetLoader:
    """Load ConceptNet CSV assertions into a KnowledgePack.

    The ConceptNet 5 CSV format has columns:
        assertion_uri, relation, subject, object, metadata_json

    Usage::

        loader = ConceptNetLoader()
        pack = loader.load_from_csv("conceptnet_assertions.csv",
                                    max_concepts=50_000, min_weight=2.0)
    """

    def __init__(self) -> None:
        self._relation_map = dict(RELATION_MAP)

    def load_from_csv(
        self,
        path_or_text: str,
        max_concepts: int = 50_000,
        min_weight: float = 2.0,
        relations: Optional[List[str]] = None,
        from_string: bool = False,
    ) -> "KnowledgePack":
        """Load a KnowledgePack from a ConceptNet CSV file (or CSV string).

        Args:
            path_or_text: Path to CSV file, or CSV text if from_string=True.
            max_concepts: Maximum number of concepts (top by degree).
            min_weight: Minimum edge weight to include.
            relations: ConceptNet relation names to include (None = use defaults).
            from_string: If True, treat path_or_text as CSV text.

        Returns:
            KnowledgePack populated with concepts, relations, and causal links.
        """
        from python.core.integration.knowledge_pack import KnowledgePack

        allowed_rels = set(relations or _DEFAULT_RELATIONS)

        # Degree counter for top-N concept selection
        degree: Dict[str, int] = defaultdict(int)
        rows: List[tuple] = []

        def _iter_rows(reader_obj: Any) -> None:
            for row in reader_obj:
                if len(row) < 5:
                    continue
                _uri, rel_uri, subj_uri, obj_uri, meta = row[:5]
                rel_parts = rel_uri.split("/")
                rel_name = rel_parts[-1] if rel_parts else ""
                if rel_name not in allowed_rels:
                    continue
                subj = _en_concept(subj_uri)
                obj = _en_concept(obj_uri)
                if subj is None or obj is None:
                    continue
                try:
                    meta_dict = json.loads(meta)
                    weight = float(meta_dict.get("weight", 1.0))
                except Exception:
                    weight = 1.0
                if weight < min_weight:
                    continue
                rows.append((rel_name, subj, obj, weight))
                degree[subj] += 1
                degree[obj] += 1

        if from_string:
            _iter_rows(csv.reader(io.StringIO(path_or_text), delimiter="\t"))
        else:
            with open(path_or_text, "r", encoding="utf-8", newline="") as f:
                _iter_rows(csv.reader(f, delimiter="\t"))

        # Select top max_concepts by degree
        top_concepts: set = set()
        for concept, _ in sorted(degree.items(), key=lambda x: -x[1])[:max_concepts]:
            top_concepts.add(concept)

        pack = KnowledgePack(name="conceptnet_en")
        added_concepts: set = set()
        causal_links_added: set = set()

        for rel_name, subj, obj, weight in rows:
            if subj not in top_concepts or obj not in top_concepts:
                continue
            nsck_rel = self._relation_map.get(rel_name, rel_name.lower())

            # Add concepts
            for concept in (subj, obj):
                if concept not in added_concepts:
                    pack.add_concept(concept, {"source": "conceptnet"})
                    added_concepts.add(concept)

            # Add relation
            pack.add_relation(subj, nsck_rel, obj)

            # Causal links for "Causes" relations
            if rel_name == "Causes":
                key = (subj, obj)
                if key not in causal_links_added:
                    strength = min(1.0, weight / 5.0)
                    pack.add_causal_link(subj, obj, strength=strength)
                    causal_links_added.add(key)

        logger.info(
            "ConceptNetLoader: loaded %d concepts, %d relations, %d causal links",
            len(added_concepts),
            len(pack._relations),
            len(pack._causal_links),
        )
        return pack
