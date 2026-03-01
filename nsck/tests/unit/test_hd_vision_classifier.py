"""Tests for NSCKHDVisionClassifier (V22 accuracy upgrade)."""
from __future__ import annotations

import numpy as np
import pytest


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_brightness_images(n_classes: int = 5, n_per_class: int = 60,
                            size: int = 16, seed: int = 0):
    """Return (images, labels) where each class has a distinct brightness."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    step = 200 // n_classes
    for lbl in range(n_classes):
        bright = 30 + lbl * step
        for _ in range(n_per_class):
            img = rng.uniform(bright - 15, bright + 15, (size, size)).clip(0, 255)
            images.append(img.astype(np.float64))
            labels.append(lbl)
    return images, labels


def _make_color_images(n_classes: int = 6, n_per_class: int = 60,
                       size: int = 16, seed: int = 1):
    """Return (images, labels) where each class is a distinct hue."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for lbl in range(n_classes):
        hue = lbl / n_classes
        # Simple HSV→RGB conversion for uniform hue patches
        hi = int(hue * 6) % 6
        f = hue * 6 - int(hue * 6)
        s, v = 0.9, 0.9 * 255
        p, q2, t2 = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        rgb = [(v, t2, p), (q2, v, p), (p, v, t2),
               (p, q2, v), (t2, p, v), (v, p, q2)][hi]
        base = np.array(list(rgb))
        for _ in range(n_per_class):
            img = np.clip(
                np.tile(base.reshape(1, 1, 3), (size, size, 1))
                + rng.normal(0, 12, (size, size, 3)),
                0, 255,
            )
            images.append(img.astype(np.float64))
            labels.append(lbl)
    return images, labels


# ── _extract_fixed_features ────────────────────────────────────────────────────

class TestExtractFixedFeatures:
    def test_returns_497_for_small_grey(self):
        from python.core.adapters.image_adapter import _extract_fixed_features
        img = np.random.rand(8, 8) * 255
        assert _extract_fixed_features(img).shape == (497,)

    def test_returns_497_for_large_grey(self):
        from python.core.adapters.image_adapter import _extract_fixed_features
        img = np.random.rand(224, 224) * 255
        assert _extract_fixed_features(img).shape == (497,)

    def test_returns_497_for_small_color(self):
        from python.core.adapters.image_adapter import _extract_fixed_features
        img = np.random.rand(16, 16, 3) * 255
        assert _extract_fixed_features(img).shape == (497,)

    def test_returns_497_for_large_color(self):
        from python.core.adapters.image_adapter import _extract_fixed_features
        img = np.random.rand(224, 224, 3) * 255
        assert _extract_fixed_features(img).shape == (497,)

    def test_all_values_in_0_1(self):
        from python.core.adapters.image_adapter import _extract_fixed_features
        img = np.random.rand(32, 32, 3) * 255
        f = _extract_fixed_features(img)
        assert f.min() >= -1e-9 and f.max() <= 1.0 + 1e-9


# ── _ExactVoteAccumulator ──────────────────────────────────────────────────────

class TestExactVoteAccumulator:
    def test_single_hv_round_trip(self):
        from python.core.vsa.hypervec_shim import HyperVector
        from python.core.vision.hd_classifier import _ExactVoteAccumulator
        hv = HyperVector(42)
        acc = _ExactVoteAccumulator()
        acc.add(hv)
        assert hv.similarity(acc.to_hv()) == pytest.approx(1.0, abs=1e-6)

    def test_majority_vote(self):
        from python.core.vsa.hypervec_shim import HyperVector
        from python.core.vision.hd_classifier import _ExactVoteAccumulator
        hv_a = HyperVector(100)
        hv_b = HyperVector(999)
        acc = _ExactVoteAccumulator()
        acc.add(hv_a); acc.add(hv_a); acc.add(hv_b)
        proto = acc.to_hv()
        assert proto.similarity(hv_a) > proto.similarity(hv_b)

    def test_reset(self):
        from python.core.vsa.hypervec_shim import HyperVector
        from python.core.vision.hd_classifier import _ExactVoteAccumulator
        acc = _ExactVoteAccumulator()
        acc.add(HyperVector(1))
        acc.reset()
        assert acc._n == 0
        assert acc._bit_sums is None


# ── NSCKHDVisionClassifier basic API ──────────────────────────────────────────

class TestNSCKHDVisionClassifierAPI:
    @pytest.fixture
    def trained_clf(self):
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        images, labels = _make_brightness_images(n_classes=3, n_per_class=40)
        clf = NSCKHDVisionClassifier()
        clf.fit(images, labels)
        return clf, images, labels

    def test_fit_sets_fitted(self, trained_clf):
        clf, _, _ = trained_clf
        assert clf._fitted

    def test_predict_one_returns_valid_label(self, trained_clf):
        clf, images, labels = trained_clf
        pred = clf.predict_one(images[0])
        assert pred in set(labels)

    def test_predict_returns_list_of_correct_length(self, trained_clf):
        clf, images, labels = trained_clf
        preds = clf.predict(images[:10])
        assert len(preds) == 10

    def test_predict_with_scores_returns_dict(self, trained_clf):
        clf, images, _ = trained_clf
        pred, scores = clf.predict_with_scores(images[0])
        assert isinstance(scores, dict)
        assert pred in scores

    def test_score_in_unit_interval(self, trained_clf):
        clf, images, labels = trained_clf
        acc = clf.score(images, labels)
        assert 0.0 <= acc <= 1.0

    def test_get_class_hv_returns_hv(self, trained_clf):
        clf, _, labels = trained_clf
        hv = clf.get_class_hv(labels[0])
        assert hv is not None

    def test_encode_image_hv_returns_hv(self, trained_clf):
        clf, images, _ = trained_clf
        hv = clf.encode_image_hv(images[0])
        assert hv is not None

    def test_requires_fit_before_predict(self):
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        clf = NSCKHDVisionClassifier()
        with pytest.raises(RuntimeError):
            clf.predict_one(np.zeros((8, 8)))

    def test_mixed_image_sizes_and_modes(self):
        """Handles greyscale and color images of different sizes in one fit."""
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        rng = np.random.default_rng(77)
        images = [
            rng.uniform(50, 100, (8, 8)).astype(np.float64),    # grey 8×8
            rng.uniform(150, 200, (32, 32)).astype(np.float64),  # grey 32×32
            rng.uniform(50, 100, (16, 16, 3)).astype(np.float64), # color 16×16
        ] * 20
        labels = ([0] * 20) + ([1] * 20) + ([2] * 20)
        clf = NSCKHDVisionClassifier()
        clf.fit(images, labels)
        assert clf._fitted
        pred = clf.predict_one(rng.uniform(50, 100, (8, 8)))
        assert pred in {0, 1, 2}


# ── NSCKHDVisionClassifier accuracy ───────────────────────────────────────────

class TestNSCKHDVisionClassifierAccuracy:
    """Accuracy tests on structured synthetic datasets (target ≥0.96)."""

    def test_greyscale_brightness_classes(self):
        from sklearn.model_selection import train_test_split
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        images, labels = _make_brightness_images(
            n_classes=5, n_per_class=80, size=16, seed=10
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            images, labels, test_size=0.25, random_state=42, stratify=labels
        )
        clf = NSCKHDVisionClassifier()
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert acc >= 0.96, f"Expected ≥0.96, got {acc:.4f}"

    def test_color_hue_classes(self):
        from sklearn.model_selection import train_test_split
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        images, labels = _make_color_images(
            n_classes=6, n_per_class=80, size=16, seed=20
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            images, labels, test_size=0.25, random_state=42, stratify=labels
        )
        clf = NSCKHDVisionClassifier()
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert acc >= 0.96, f"Expected ≥0.96, got {acc:.4f}"

    def test_bw_texture_classes(self):
        """Striped / checkerboard / gradient textures (greyscale 32×32)."""
        from sklearn.model_selection import train_test_split
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        rng = np.random.default_rng(30)
        images, labels = [], []
        size = 32
        for pid in range(4):
            for _ in range(80):
                img = np.zeros((size, size))
                if pid == 0:
                    for i in range(0, size, 4):
                        img[i:i+2, :] = 1.0
                elif pid == 1:
                    for j in range(0, size, 4):
                        img[:, j:j+2] = 1.0
                elif pid == 2:
                    img = ((np.arange(size).reshape(-1, 1) // 4
                            + np.arange(size) // 4) % 2).astype(float)
                elif pid == 3:
                    img = np.tile(np.linspace(0, 1, size), (size, 1))
                img = (img + rng.normal(0, 0.07, (size, size))).clip(0, 1) * 255
                images.append(img.astype(np.float64))
                labels.append(pid)
        X_tr, X_te, y_tr, y_te = train_test_split(
            images, labels, test_size=0.25, random_state=42, stratify=labels
        )
        clf = NSCKHDVisionClassifier()
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert acc >= 0.96, f"Expected ≥0.96, got {acc:.4f}"


# ── Lifelong learning: no forgetting ──────────────────────────────────────────

class TestLifelongLearning:
    def test_add_class_does_not_degrade_existing(self):
        """add_class for new label must not change existing class centroids."""
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        rng = np.random.default_rng(99)

        # Phase 1: train on 3 classes
        images_p1, labels_p1 = _make_brightness_images(
            n_classes=3, n_per_class=80, size=16, seed=5
        )
        clf = NSCKHDVisionClassifier()
        clf.fit(images_p1, labels_p1)

        # Evaluate phase-1 accuracy before adding anything
        test_imgs_p1 = [rng.uniform(30 + c * 40 - 15, 30 + c * 40 + 15,
                                    (16, 16)).clip(0, 255).astype(np.float64)
                        for c in range(3) for _ in range(20)]
        test_lbls_p1 = [c for c in range(3) for _ in range(20)]
        acc_before = clf.score(test_imgs_p1, test_lbls_p1)

        # Add a genuinely new class (different brightness range not in phase 1)
        new_imgs = [rng.uniform(220, 240, (16, 16)).clip(0, 255).astype(np.float64)
                    for _ in range(40)]
        clf.add_class(99, new_imgs)

        # Phase-1 class accuracy must not drop
        acc_after = clf.score(test_imgs_p1, test_lbls_p1)
        assert acc_after >= acc_before, (
            f"Forgetting: acc dropped from {acc_before:.3f} to {acc_after:.3f}"
        )

    def test_new_class_is_classifiable(self):
        """After add_class, the new class can be predicted."""
        from python.core.vision.hd_classifier import NSCKHDVisionClassifier
        images, labels = _make_brightness_images(
            n_classes=3, n_per_class=60, size=16, seed=6
        )
        clf = NSCKHDVisionClassifier()
        clf.fit(images, labels)

        # New class with very bright images
        new_imgs = [np.full((16, 16), 240.0) + np.random.default_rng(i).normal(0, 3, (16, 16))
                    for i in range(40)]
        clf.add_class("bright", new_imgs)

        # Query a bright image → must predict "bright"
        q = np.full((16, 16), 240.0)
        pred = clf.predict_one(q)
        assert pred == "bright", f"Expected 'bright', got {pred!r}"
