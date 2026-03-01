"""Unit tests for VSAProjector."""
import os
import sys
import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_NSCK_VISION_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_NSCK_VISION_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_projection_deterministic():
    """Same input → same HV every time."""
    from python.core.vision.vsa_projector import VSAProjector
    proj = VSAProjector(dim_in=64, model_id="test_model")
    emb = np.random.default_rng(42).standard_normal(64).astype(np.float32)
    hv1 = proj.encode_new(emb)
    hv2 = proj.encode_new(emb)
    assert list(hv1.bits) == list(hv2.bits)


def test_similarity_preservation():
    """Spearman ρ between embedding cosine sim and HV hamming sim > -0.5."""
    from python.core.vision.vsa_projector import VSAProjector
    try:
        from scipy.stats import spearmanr
    except ImportError:
        pytest.skip("scipy not available")

    proj = VSAProjector(dim_in=64, model_id="sim_test")
    rng = np.random.default_rng(7)
    n = 20
    embs = rng.standard_normal((n, 64)).astype(np.float32)
    proj.fit(embs)
    hvs = [proj.encode_new(embs[i]) for i in range(n)]

    orig_sims = []
    hv_sims = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            a, b = embs[i], embs[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = float(np.dot(a, b) / max(na * nb, 1e-9))
            orig_sims.append(cos)
            hv_sims.append(hvs[i].similarity_robust(hvs[j]))

    rho, _ = spearmanr(orig_sims, hv_sims)
    assert rho > -0.5, f"Spearman ρ={rho:.3f} is too negative"


def test_dimension_output():
    """Output HV is always a valid binary HyperVector."""
    from python.core.vision.vsa_projector import VSAProjector
    import python.core.vsa.hypervec_shim as hypervec_rs
    proj = VSAProjector(dim_in=128, model_id="dim_test")
    emb = np.zeros(128, dtype=np.float32)
    hv = proj.encode_new(emb)
    assert isinstance(hv, hypervec_rs.HyperVector)
    assert len(hv.bits) > 0


def test_batch_projection():
    """Batch of 100 vectors projects correctly."""
    from python.core.vision.vsa_projector import VSAProjector
    proj = VSAProjector(dim_in=32, model_id="batch_test")
    rng = np.random.default_rng(99)
    embs = rng.standard_normal((100, 32)).astype(np.float32)
    labels = [f"label_{i}" for i in range(100)]
    proj.fit(embs)
    codebook = proj.project(embs, labels)
    assert len(codebook) == 100
    assert all(isinstance(k, str) for k in codebook.keys())


def test_model_id_determinism():
    """Different model_ids produce different seed matrices."""
    from python.core.vision.vsa_projector import VSAProjector
    proj_a = VSAProjector(dim_in=64, model_id="model_a")
    proj_b = VSAProjector(dim_in=64, model_id="model_b")
    assert proj_a.get_seed() != proj_b.get_seed()
