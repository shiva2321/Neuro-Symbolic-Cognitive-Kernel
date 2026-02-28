"""
KnowledgeSeeder — bootstrap a domain from a declarative YAML spec.

Usage:
    seeder = KnowledgeSeeder()
    seeder.seed_from_yaml("path/to/navigation.yaml", engine)
    # engine now has pre-populated rules, causal graph, and procedural memory
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nsck.bootstrap.knowledge_seeder")


class KnowledgeSeeder:
    """Seeds a CognitiveEngine from a declarative YAML domain kit."""

    def seed_from_yaml(self, yaml_path: str, engine) -> int:
        """
        Load a YAML domain kit and inject knowledge into engine.

        Parameters
        ----------
        yaml_path : str
            Path to the domain YAML file.
        engine : CognitiveEngine
            The engine to seed.

        Returns
        -------
        int
            Number of rules seeded.
        """
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed; cannot seed from YAML. pip install pyyaml")
            return 0

        if not os.path.exists(yaml_path):
            logger.error("Domain kit not found: %s", yaml_path)
            return 0

        with open(yaml_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        if not isinstance(spec, dict):
            logger.error("Invalid YAML format in %s", yaml_path)
            return 0

        domain = spec.get("domain", "unknown")
        logger.info("[SEEDER] Seeding domain '%s' from %s", domain, yaml_path)

        seeded_rules = 0

        # 1. Seed semantic concepts
        for concept_spec in spec.get("semantic_concepts", []):
            try:
                name = concept_spec.get("name", "")
                props = concept_spec.get("properties", {})
                if name and hasattr(engine, "semantic_memory"):
                    engine.semantic_memory.add_concept(name, props)
            except Exception as exc:
                logger.debug("[SEEDER] Concept seeding failed for %s: %s", concept_spec, exc)

        # 2. Seed causal graph entries
        causal_graph = None
        if hasattr(engine, "causal_graphs"):
            if domain not in engine.causal_graphs:
                from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner
                engine.causal_graphs[domain] = CausalGraph()
                engine.causal_reasoners[domain] = CausalReasoner(engine.causal_graphs[domain])
            causal_graph = engine.causal_graphs.get(domain)

        for edge in spec.get("causal_graph", []):
            try:
                cause = edge.get("cause", "")
                effect = edge.get("effect", "")
                strength = float(edge.get("strength", 0.5))
                if cause and effect and causal_graph is not None:
                    causal_graph.add_causes(cause, effect, strength)
            except Exception as exc:
                logger.debug("[SEEDER] Causal edge seeding failed: %s", exc)

        # 3. Seed causal rules
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.integration.persistence import Rule

        for rule_spec in spec.get("causal_rules", []):
            try:
                conditions = frozenset(rule_spec.get("condition", []))
                action = rule_spec.get("action", "")
                outcome = rule_spec.get("outcome", "neutral")
                confidence = float(rule_spec.get("confidence", 0.7))
                support = int(rule_spec.get("support", 5))
                if not action:
                    continue

                rule = Rule(
                    id=None,
                    condition=conditions,
                    consequence=action,
                    source="bootstrap",
                    task_tag=domain,
                    support_count=support,
                    success_rate=confidence,
                    confidence=confidence,
                )

                # Inject into rule learner
                if hasattr(engine, "rule_learner"):
                    if domain not in engine.rule_learner.learned_rules:
                        engine.rule_learner.learned_rules[domain] = []
                    engine.rule_learner.learned_rules[domain].append(rule)
                    seeded_rules += 1

                # For high-confidence rules, also cache in procedural memory
                if confidence >= 0.8 and hasattr(engine, "procedural_memory"):
                    # Build a synthetic HV for the condition set
                    cond_hv = hypervec_rs.HyperVector(hash(str(sorted(conditions))) % (2**32))
                    engine.procedural_memory.cache_skill(
                        context_hv=cond_hv,
                        action=action,
                        reward=confidence,
                        label=domain,
                    )

            except Exception as exc:
                logger.debug("[SEEDER] Rule seeding failed: %s", exc)

        logger.info("[SEEDER] Domain '%s' seeded: %d rules", domain, seeded_rules)
        return seeded_rules
