"""SemanticSeeder — orchestrates ConceptNet + BERT seeding."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("nsck.seeding.semantic_seeder")


class SemanticSeeder:
    """High-level orchestrator for seeding an NSCKSubstrate.

    Usage::

        seeder = SemanticSeeder()
        seeder.full_seed(substrate, conceptnet_pack="data/cn.kp")
    """

    def seed_from_conceptnet_pack(
        self,
        substrate: Any,
        pack_path: str,
    ) -> Dict[str, int]:
        """Load a ConceptNet KnowledgePack and inject into substrate.

        Returns:
            Dict with counts: {"concepts": N, "relations": M, "causal_links": K}
        """
        from python.core.integration.knowledge_pack import KnowledgePack

        pack = KnowledgePack.load(pack_path)
        engine = getattr(substrate, "engine", substrate)
        counts = pack.inject_into(engine)
        logger.info("seed_from_conceptnet_pack: %s", counts)
        return counts

    def seed_from_bert(
        self,
        substrate: Any,
        model_name: str = "bert-base-uncased",
        save_pack: Optional[str] = None,
    ) -> Any:
        """Seed substrate with BERT embeddings.

        Returns:
            TransplantReport from BertSeeder.
        """
        from python.core.seeding.bert_seeder import BertSeeder

        seeder = BertSeeder()
        return seeder.seed(substrate, model_name=model_name, save_pack=save_pack)

    def post_seed_enrich(self, substrate: Any) -> None:
        """Run post-seeding enrichment: transitive closure + prototypes."""
        engine = getattr(substrate, "engine", substrate)
        sem = getattr(engine, "semantic_memory", None)
        if sem is None:
            return
        # Transitive closure on is_a
        try:
            sem.infer_transitive("is_a")
            logger.info("post_seed_enrich: is_a transitive closure done")
        except AttributeError:
            try:
                sem.infer_transitive_closure("is_a")
            except AttributeError:
                logger.debug("infer_transitive not available")
        # Build prototypes
        try:
            sem.build_prototypes()
            logger.info("post_seed_enrich: prototypes built")
        except AttributeError:
            logger.debug("build_prototypes not available")

    def full_seed(
        self,
        substrate: Any,
        conceptnet_pack: Optional[str] = None,
        bert_model: Optional[str] = None,
        save_fused_pack: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full seeding pipeline.

        Args:
            substrate: NSCKSubstrate.
            conceptnet_pack: Path to a .kp file with ConceptNet data.
            bert_model: HuggingFace model name (optional; skipped if None).
            save_fused_pack: Path to save fused KnowledgePack (optional).

        Returns:
            Dict with seeding results.
        """
        results: Dict[str, Any] = {}

        if conceptnet_pack:
            results["conceptnet"] = self.seed_from_conceptnet_pack(substrate, conceptnet_pack)

        if bert_model:
            try:
                results["bert"] = self.seed_from_bert(substrate, model_name=bert_model)
            except RuntimeError as exc:
                logger.warning("BERT seeding skipped: %s", exc)
                results["bert"] = None

        self.post_seed_enrich(substrate)
        results["enriched"] = True

        if save_fused_pack is not None:
            try:
                from python.core.integration.knowledge_pack import KnowledgePack
                engine = getattr(substrate, "engine", substrate)
                sem = getattr(engine, "semantic_memory", None)
                if sem is not None:
                    fused = KnowledgePack(name="fused_seed")
                    for name in list(getattr(sem, 'concept_hvs', getattr(sem, 'concepts', {})).keys())[:1000]:
                        fused.add_concept(name, {})
                    fused.save(save_fused_pack)
                    logger.info("Saved fused pack to %s", save_fused_pack)
            except Exception as exc:
                logger.warning("Failed to save fused pack: %s", exc)

        return results
