"""BERT → NSCK transplant seeder."""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger("nsck.seeding.bert_seeder")


class BertSeeder:
    """Seed an NSCKSubstrate with BERT embeddings via the transplant pipeline.

    Requires the ``transformers`` package.  If not installed a clear
    ``RuntimeError`` is raised.

    Usage::

        seeder = BertSeeder()
        report = seeder.seed(substrate, model_name="bert-base-uncased")
    """

    def seed(
        self,
        substrate: Any,
        model_name: str = "bert-base-uncased",
        save_pack: Optional[str] = None,
    ) -> Any:
        """Seed substrate with BERT embeddings.

        Args:
            substrate: NSCKSubstrate instance.
            model_name: HuggingFace model name.
            save_pack: Optional path to save the resulting KnowledgePack.

        Returns:
            TransplantReport from the transplant pipeline.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
        """
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for BERT seeding. "
                "Install it with: pip install transformers"
            ) from exc

        report = substrate.transplant(model_name, domain_name="language", strategy="svd_factored")

        if save_pack is not None:
            try:
                from python.core.integration.knowledge_pack import KnowledgePack
                pack = KnowledgePack(name=f"bert_{model_name}")
                pack.save(save_pack)
                logger.info("Saved BERT knowledge pack to %s", save_pack)
            except Exception as exc:
                logger.warning("Failed to save BERT pack: %s", exc)

        return report
