"""End-to-end integration test for the transplantation pipeline (V15).

Uses synthetic embeddings with known cluster structure to verify the full
pipeline produces usable knowledge in NSCK.
"""
from __future__ import annotations

import sys
import os
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Mock model factory
# ---------------------------------------------------------------------------

class _FakeParam:
    def __init__(self, arr):
        self._arr = arr.astype(np.float32)
    @property
    def data(self): return self
    def detach(self): return self
    def numpy(self): return self._arr
    @property
    def shape(self): return self._arr.shape
    @property
    def ndim(self): return self._arr.ndim


def _make_cluster_model(n_clusters=5, per_cluster=20, dim=32, seed=0):
    """Build a mock model whose embeddings have clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 5.0
    emb = np.vstack([
        centers[k] + rng.standard_normal((per_cluster, dim)) * 0.05
        for k in range(n_clusters)
    ]).astype(np.float32)

    class _WE:
        pass

    class _Model:
        def __init__(self, e):
            self.word_embeddings = _WE()
            self.word_embeddings.weight = _FakeParam(e)
        def named_parameters(self):
            yield "word_embeddings.weight", self.word_embeddings.weight

    return _Model(emb), emb, {f"token_{i}": i for i in range(len(emb))}


def _make_structured_model():
    """Build a mock model with king/queen/man/woman analogy structure."""
    rng = np.random.default_rng(42)
    d = 64

    def _norm(v):
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    base = _norm(rng.standard_normal(d).astype(np.float32))
    royalty = _norm(rng.standard_normal(d).astype(np.float32))
    male = _norm(rng.standard_normal(d).astype(np.float32))
    female = _norm(rng.standard_normal(d).astype(np.float32))
    animal = _norm(rng.standard_normal(d).astype(np.float32))

    # king/queen/man/woman
    king  = _norm(base * 3 + royalty * 3 + male * 3)
    queen = _norm(base * 3 + royalty * 3 + female * 3)
    man   = _norm(base * 3 + male * 3)
    woman = _norm(base * 3 + female * 3)

    # Animal cluster
    cat   = _norm(animal * 3 + rng.standard_normal(d).astype(np.float32) * 0.1)
    dog   = _norm(animal * 3 + rng.standard_normal(d).astype(np.float32) * 0.1)
    bird  = _norm(animal * 3 + rng.standard_normal(d).astype(np.float32) * 0.1)
    lion  = _norm(animal * 3 + rng.standard_normal(d).astype(np.float32) * 0.1)
    eagle = _norm(animal * 3 + rng.standard_normal(d).astype(np.float32) * 0.1)

    # Filler items (random)
    fillers = rng.standard_normal((90, d)).astype(np.float32)
    fillers = fillers / np.maximum(np.linalg.norm(fillers, axis=1, keepdims=True), 1e-9)

    named_order = ["king", "queen", "man", "woman", "cat", "dog", "bird", "lion", "eagle"]
    named_emb = np.stack([king, queen, man, woman, cat, dog, bird, lion, eagle], axis=0)
    all_emb = np.vstack([named_emb, fillers])
    token_names = named_order + [f"filler_{i}" for i in range(len(fillers))]
    vm = {t: i for i, t in enumerate(token_names)}

    class _WE:
        pass

    class _Model:
        def __init__(self):
            self.word_embeddings = _WE()
            self.word_embeddings.weight = _FakeParam(all_emb)
        def named_parameters(self):
            yield "word_embeddings.weight", self.word_embeddings.weight

    return _Model(), all_emb, vm, token_names


def _hamming_sim(b1: np.ndarray, b2: np.ndarray) -> float:
    return 2.0 * float((b1 == b2).mean()) - 1.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2ETransplantPipeline:

    def test_pipeline_runs_end_to_end(self):
        """Full pipeline should run without error and return a TransplantReport."""
        model, emb, vm = _make_cluster_model(n_clusters=5, per_cluster=10, dim=16)
        from python.core.transplant.pipeline import TransplantPipeline
        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="language",
            strategy="svd_factored",
            calibration_epochs=0,
        )
        assert report is not None
        assert report.n_concepts == 50

    def test_cluster_preservation_random_projector(self):
        """Random projection should preserve cluster structure (ρ >= 0.50)."""
        model, emb, vm = _make_cluster_model(n_clusters=5, per_cluster=20, dim=32, seed=7)
        n = len(emb)
        from python.core.transplant.projector import RandomProjector
        from python.core.transplant.validator import TransplantValidator
        cb = RandomProjector(32, seed=0).project(emb, vm)
        v = TransplantValidator(rho_threshold=0.0, recall10_threshold=0.0,
                                recall50_threshold=0.0, ari_threshold=0.0)
        report = v.validate(emb, cb, vm, n_sample_pairs=500)
        assert report.spearman_rho >= 0.50, (
            f"RandomProjector ρ={report.spearman_rho:.3f} below 0.50"
        )

    def test_cluster_preservation_svd_projector(self):
        """SVD+FPE should preserve coarse cluster structure (intra > inter similarity)."""
        _, emb, vm, _ = _make_structured_model()
        from python.core.transplant.projector import SVDFactoredProjector
        cb = SVDFactoredProjector(emb.shape[1], n_components=16, n_bins=8, seed=0).project(emb, vm)

        D = 10240
        animals = ["cat", "dog", "bird", "lion", "eagle"]
        # Build bits for animals and humans
        animal_bits = [cb[a].bits[:D].astype(np.int32) for a in animals]
        human_bits = [cb[h].bits[:D].astype(np.int32) for h in ["king", "queen", "man", "woman"]]

        # Mean sim within animals
        intra = []
        for i in range(len(animals)):
            for j in range(i + 1, len(animals)):
                intra.append(_hamming_sim(animal_bits[i], animal_bits[j]))
        # Mean sim animal→human
        cross = []
        for ab in animal_bits:
            for hb in human_bits:
                cross.append(_hamming_sim(ab, hb))

        mean_intra = float(np.mean(intra))
        mean_cross = float(np.mean(cross))
        assert mean_intra > mean_cross, (
            f"Animal cluster sim ({mean_intra:.3f}) should exceed "
            f"human cross-cluster sim ({mean_cross:.3f})"
        )

    def test_semantic_memory_injection(self):
        """After pipeline with relaxed thresholds, concepts inject into semantic memory."""
        model, emb, vm = _make_cluster_model(n_clusters=3, per_cluster=5, dim=8)
        from python.core.transplant.pipeline import TransplantPipeline
        from python.core.transplant.validator import TransplantValidator

        pipe = TransplantPipeline()
        pipe._validator = TransplantValidator(
            rho_threshold=-1.0, recall10_threshold=-1.0,
            recall50_threshold=-1.0, ari_threshold=-1.0,
        )

        class _MockMemory:
            def __init__(self):
                self.concept_hvs = {}
                self._added = []
            def add_concept(self, name, props):
                self._added.append(name)
            def add_relation(self, *_): pass

        class _MockEngine:
            def __init__(self): self.semantic_memory = _MockMemory()

        engine = _MockEngine()
        pipe.run(model, domain_name="inject", strategy="random",
                 calibration_epochs=0, cognitive_engine=engine)
        assert len(engine.semantic_memory._added) == 15  # 3 × 5

    def test_knowledge_pack_save_load_roundtrip(self):
        """Saving and loading a KnowledgePack preserves all concepts."""
        model, emb, vm = _make_cluster_model(n_clusters=2, per_cluster=5, dim=8)
        from python.core.transplant.pipeline import TransplantPipeline
        from python.core.integration.knowledge_pack import KnowledgePack

        pipe = TransplantPipeline()
        with tempfile.NamedTemporaryFile(suffix=".kp", delete=False) as f:
            path = f.name
        try:
            pipe.run(model, domain_name="roundtrip", strategy="random",
                     calibration_epochs=0, save_pack_path=path)
            pack = KnowledgePack.load(path)
            assert pack.name == "roundtrip"
            # All tokens should be in the pack (harvester names them token_0 etc.)
            saved_names = {name for name, _, _ in pack._concepts}
            assert len(saved_names) == 10  # 2 clusters × 5
            for i in range(10):
                assert f"token_{i}" in saved_names
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_full_pipeline_with_calibration(self):
        """Full pipeline including STDP calibration should complete."""
        model, emb, vm = _make_cluster_model(n_clusters=2, per_cluster=5, dim=8)
        from python.core.transplant.pipeline import TransplantPipeline

        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="calibrated",
            strategy="random",
            calibration_epochs=1,
        )
        assert report is not None
        assert report.calibration_quality_curve is not None

    def test_recall_at_50_achievable_with_random_projector(self):
        """Random projector should achieve Recall@50 >= 0.40 on tight clusters."""
        model, emb, vm = _make_cluster_model(n_clusters=5, per_cluster=30, dim=32, seed=1)
        from python.core.transplant.projector import RandomProjector
        from python.core.transplant.validator import TransplantValidator
        cb = RandomProjector(32, seed=0).project(emb, vm)
        v = TransplantValidator(rho_threshold=0.0, recall10_threshold=0.0,
                                recall50_threshold=0.0, ari_threshold=0.0)
        report = v.validate(emb, cb, vm, n_sample_pairs=500)
        assert report.recall_at_50 >= 0.40, (
            f"RandomProjector Recall@50={report.recall_at_50:.3f} below 0.40"
        )
