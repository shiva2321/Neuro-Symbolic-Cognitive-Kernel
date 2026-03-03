"""Model Transplantation Pipeline — orchestrates Harvest → Project → Calibrate → Validate → Integrate."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from python.core.transplant.harvester import ModelHarvester, HarvestResult
from python.core.transplant.projector import (
    BaseProjector,
    RandomProjector,
    LearnedProjector,
    SVDFactoredProjector,
)
from python.core.transplant.calibrator import STDPCalibrator, CalibratedResult
from python.core.transplant.validator import TransplantValidator, TransplantReport

_log = logging.getLogger(__name__)

# Maximum number of tokens to consider for pairwise similarity-based relation
# injection.  Caps the O(n²) cost of the relation-addition step.
_RELATION_TOKEN_CAP = 500


class TransplantPipeline:
    """End-to-end model transplantation pipeline.

    Stages
    ------
    1. **Harvest**   — extract embedding matrix from source model.
    2. **Project**   — map embeddings to HyperVectors.
    3. **Calibrate** — optional STDP fine-tuning.
    4. **Validate**  — measure neighbourhood-preservation quality.
    5. **Integrate** — inject into cognitive engine's semantic memory.
    6. **Save**      — persist as a :class:`~python.core.integration.knowledge_pack.KnowledgePack`.

    Parameters
    ----------
    config:
        Optional NSCK config object.  Fields ``transplant_rho_threshold``,
        ``transplant_recall10_threshold``, ``transplant_recall50_threshold``,
        and ``transplant_ari_threshold`` are read if present.
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config
        # Cache of fitted projectors keyed by domain_name
        self._projectors: Dict[str, BaseProjector] = {}
        # Cache of codebooks by domain name (used by societal_transplant)
        self._codebooks: Dict[str, Any] = {}

        # Read thresholds from config with sensible defaults
        def _cfg(attr: str, default: float) -> float:
            return float(getattr(config, attr, default)) if config is not None else default

        self._validator = TransplantValidator(
            rho_threshold=_cfg("transplant_rho_threshold", 0.80),
            recall10_threshold=_cfg("transplant_recall10_threshold", 0.70),
            recall50_threshold=_cfg("transplant_recall50_threshold", 0.60),
            ari_threshold=_cfg("transplant_ari_threshold", 0.65),
        )

    # ------------------------------------------------------------------

    def run(
        self,
        model: Any,
        domain_name: str,
        strategy: str = "svd_factored",
        calibration_epochs: Optional[int] = None,
        save_pack_path: Optional[str] = None,
        cognitive_engine: Any = None,
    ) -> TransplantReport:
        """Execute the full transplantation pipeline.

        Parameters
        ----------
        model:
            Source neural network model (PyTorch or duck-typed).
        domain_name:
            Human-readable label for this knowledge domain.
        strategy:
            Projection strategy: ``"random"``, ``"learned"``, or
            ``"svd_factored"`` (default).
        calibration_epochs:
            Number of STDP calibration epochs.  ``0`` or ``None`` skips
            calibration.
        save_pack_path:
            If provided, save the resulting ``KnowledgePack`` here.
        cognitive_engine:
            If provided and validation passes, inject concepts into its
            semantic memory.

        Returns
        -------
        TransplantReport
        """
        # 1. HARVEST
        harvest: HarvestResult = ModelHarvester().harvest(model)
        if harvest.model_type == "error":
            raise ValueError(
                f"Model harvesting failed: {harvest.metadata.get('error', 'unknown error')}"
            )

        # 2. PROJECT
        projector = self._make_projector(strategy, harvest.embedding_dim)
        codebook = projector.project(harvest.embeddings, harvest.vocab_mapping)
        self._projectors[domain_name] = projector
        self._codebooks[domain_name] = codebook

        # 3. CALIBRATE (optional)
        cal_result: Optional[CalibratedResult] = None
        n_cal_epochs = calibration_epochs if calibration_epochs is not None else 0
        if n_cal_epochs > 0:
            calibrator = STDPCalibrator(
                input_dim=harvest.embedding_dim,
                n_epochs=n_cal_epochs,
            )
            cal_result = calibrator.calibrate(
                harvest.embeddings, codebook, harvest.vocab_mapping
            )
            codebook = cal_result.codebook

        # 4. VALIDATE
        report = self._validator.validate(
            harvest.embeddings, codebook, harvest.vocab_mapping
        )

        # Annotate report with optional extra info
        if cal_result is not None:
            report.calibration_quality_curve = cal_result.quality_curve

        # 5. INTEGRATE
        if report.passed and cognitive_engine is not None:
            self._integrate(codebook, harvest, cognitive_engine)

        # 6. SAVE
        if save_pack_path is not None:
            self._save_pack(codebook, harvest, domain_name, save_pack_path)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_projector(strategy: str, embedding_dim: int) -> BaseProjector:
        if strategy == "random":
            return RandomProjector(embedding_dim)
        elif strategy == "learned":
            return LearnedProjector(embedding_dim)
        elif strategy == "svd_factored":
            return SVDFactoredProjector(embedding_dim)
        else:
            raise ValueError(
                f"Unknown projection strategy {strategy!r}. "
                "Choose 'random', 'learned', or 'svd_factored'."
            )

    @staticmethod
    def _integrate(
        codebook: Dict[str, Any],
        harvest: HarvestResult,
        cognitive_engine: Any,
    ) -> None:
        """Inject concepts and high-similarity relations into semantic memory."""
        sem = getattr(cognitive_engine, "semantic_memory", None)
        if sem is None:
            return

        # Inject all concepts
        for token, hv in codebook.items():
            try:
                idx = harvest.vocab_mapping.get(token)
                props = {"domain": harvest.model_type, "token": token}
                if idx is not None:
                    props["embedding_index"] = idx
                sem.add_concept(token, props)
                if hasattr(sem, "concept_hvs"):
                    sem.concept_hvs[token] = hv
            except Exception as exc:  # noqa: BLE001
                _log.debug("Failed to inject concept %r: %s", token, exc)

        # Add relations between highly similar concepts (similarity > 0.7)
        tokens = list(codebook.keys())
        n = len(tokens)
        for i in range(min(n, _RELATION_TOKEN_CAP)):          # cap to avoid O(n²) cost
            for j in range(i + 1, min(n, _RELATION_TOKEN_CAP)):
                try:
                    sim = codebook[tokens[i]].similarity(codebook[tokens[j]])
                    if sim > 0.7:
                        if hasattr(sem, "add_relation"):
                            sem.add_relation(tokens[i], "similar_to", tokens[j])
                except Exception as exc:  # noqa: BLE001
                    _log.debug("Failed to add relation %r → %r: %s", tokens[i], tokens[j], exc)

    @staticmethod
    def _save_pack(
        codebook: Dict[str, Any],
        harvest: HarvestResult,
        domain_name: str,
        path: str,
    ) -> None:
        from python.core.integration.knowledge_pack import KnowledgePack  # noqa: PLC0415

        pack = KnowledgePack(name=domain_name)
        for token, hv in codebook.items():
            idx = harvest.vocab_mapping.get(token)
            props: Dict[str, Any] = {
                "domain": harvest.model_type,
                "token": token,
                "source_model": harvest.source_model[:200],
            }
            if idx is not None:
                props["embedding_index"] = idx
            pack.add_concept(token, props, hv)

        # Add similarity relations (same threshold as integrate)
        tokens = list(codebook.keys())
        n = len(tokens)
        for i in range(min(n, _RELATION_TOKEN_CAP)):
            for j in range(i + 1, min(n, _RELATION_TOKEN_CAP)):
                try:
                    sim = codebook[tokens[i]].similarity(codebook[tokens[j]])
                    if sim > 0.7:
                        pack.add_relation(tokens[i], "similar_to", tokens[j])
                except Exception as exc:  # noqa: BLE001
                    _log.debug("Failed to persist relation %r → %r: %s", tokens[i], tokens[j], exc)

        pack.save(path)

    def societal_transplant(
        self,
        model: Any,
        domain_name: str,
        strategy: str = "svd_factored",
        calibration_epochs: Optional[int] = None,
        save_pack_path: Optional[str] = None,
        cognitive_engine: Any = None,
        societal_manager: Any = None,
        bond_threshold: float = 0.65,
        cluster_resolution: float = 1.0,
    ) -> "TransplantReport":
        """Transplant + inject concepts as LivingHyperVectors into a SocietyManager.

        Extends :meth:`run` by additionally:

        1. Wrapping each projected HV as a :class:`~python.core.societal.LivingHyperVector`
           in the supplied *societal_manager*.
        2. Running :meth:`~python.core.societal.SocietyManager.auto_bond` to
           form bonds between similar concepts.
        3. Running Leiden clustering at *cluster_resolution* to assign community
           membership.

        Parameters
        ----------
        societal_manager:
            A :class:`~python.core.societal.SocietyManager` instance.  If
            ``None``, a fresh one is created and returned via the report's
            ``metadata`` dict (key ``"societal_manager"``).
        bond_threshold:
            Passed to ``SocietyManager.bond_threshold`` when creating a new
            manager.
        cluster_resolution:
            Leiden γ parameter for community detection.

        Returns
        -------
        TransplantReport
            Same report as :meth:`run`, with societal metadata added to
            ``report.metadata``.
        """
        from python.core.societal.society_manager import SocietyManager
        from python.core.societal.living_hypervector import LivingHyperVector

        # Run standard transplant pipeline first
        report = self.run(
            model=model,
            domain_name=domain_name,
            strategy=strategy,
            calibration_epochs=calibration_epochs,
            save_pack_path=save_pack_path,
            cognitive_engine=cognitive_engine,
        )

        # Build / reuse SocietyManager
        if societal_manager is None:
            societal_manager = SocietyManager(bond_threshold=bond_threshold)

        # Retrieve the projected codebook from the cache
        codebook: Dict[str, Any] = self._codebooks.get(domain_name, {})
        if not codebook:
            _log.warning(
                "[SOCIETAL] No codebook found for domain %r; "
                "societal transplant skipped.", domain_name
            )
        else:
            epoch = societal_manager.epoch
            for token, hv in codebook.items():
                lhv = LivingHyperVector(
                    concept_id=token,
                    hv=hv,
                    domain_path=[domain_name],
                    role="leaf",
                    birth_epoch=epoch,
                    metadata={"source_model": domain_name},
                )
                societal_manager.register(lhv)

            # Form bonds between similar concepts
            tokens_list = list(codebook.keys())
            societal_manager.auto_bond(candidates=tokens_list, bond_type="similarity")

            # Run Leiden clustering
            cluster_result = societal_manager.leiden_cluster(cluster_resolution)
            _log.info(
                "[SOCIETAL] Leiden: %d communities, Q=%.4f for domain %r",
                cluster_result.n_communities,
                cluster_result.modularity,
                domain_name,
            )

        # Attach societal info to report metadata
        if not hasattr(report, "metadata") or report.metadata is None:
            try:
                import dataclasses as _dc
                report = _dc.replace(
                    report, metadata={"societal_manager": societal_manager}
                )
            except Exception:
                pass
        else:
            report.metadata["societal_manager"] = societal_manager

        return report
