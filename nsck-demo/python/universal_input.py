"""
NSCK Phase 8: Universal Input Layer
====================================
Maps arbitrary sensor data (scalars, categories, dicts, lists) into the
10,240-bit binary VSA hypervector space so that every modality shares
the same algebraic structure (XOR=bind, bundle=superposition, permute=sequence).

Design decisions
----------------
* **Scalars** use Thermometer Encoding (quantise into N bins → deterministic
  HVs → weighted bundle by proximity).  Preserves the *Scalar Similarity
  Test*: nearby values produce high cosine overlap.
* **Categoricals** are codebook entries with deterministic seeding per
  ``domain:label`` pair.  An LRU cache (max_size) prevents codebook explosion.
* **Dicts** use recursive role-filler binding: Role_HV ⊗ Value_HV for each
  key, then XOR-bundle all pairs.
* **Lists** use permutation-based sequence encoding: A ⊕ ρ(B) ⊕ ρ²(C).
"""

from __future__ import annotations

import hashlib
import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np
import hypervec_shim as hv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSION = 10240
DEFAULT_THERMOMETER_BINS = 100
MAX_CODEBOOK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stable_seed(label: str) -> int:
    """Deterministic 64-bit seed from an arbitrary string label."""
    h = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0xFFFF_FFFF  # u32 to match HV ctor


# ---------------------------------------------------------------------------
# Universal Input
# ---------------------------------------------------------------------------
class UniversalInput:
    """
    Grounding module that converts heterogeneous data into 10 240-bit HVs.

    Usage::

        ui = UniversalInput()
        hv_scalar  = ui.ground(0.73, domain="sensor:temperature")
        hv_cat     = ui.ground("error",  domain="status")
        hv_dict    = ui.ground({"ip": "10.0.0.1", "status": "ok"}, domain="net")
        hv_seq     = ui.ground([0.1, 0.5, 0.9], domain="trajectory")
    """

    def __init__(
        self,
        n_bins: int = DEFAULT_THERMOMETER_BINS,
        max_codebook: int = MAX_CODEBOOK_SIZE,
    ):
        self.n_bins = n_bins
        self.max_codebook = max_codebook

        # LRU codebook:  "domain:label" → HyperVector
        self._codebook: collections.OrderedDict[str, hv.HyperVector] = (
            collections.OrderedDict()
        )

        # Thermometer bin HVs (lazy-init per domain)
        # domain → list[HyperVector]  (one per bin)
        self._bin_hvs: Dict[str, list] = {}

        # Role HVs for dict keys (lazy-init)
        self._role_hvs: Dict[str, hv.HyperVector] = {}

        # Stats for telemetry
        self._stats = {
            "scalars_grounded": 0,
            "categories_grounded": 0,
            "dicts_grounded": 0,
            "lists_grounded": 0,
            "codebook_size": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ground(
        self,
        data: Any,
        domain: str = "default",
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> hv.HyperVector:
        """Auto-detect type and ground *data* into a hypervector.

        Parameters
        ----------
        data : scalar | str | dict | list
        domain : namespacing tag (e.g. ``"sensor:temperature"``)
        min_val, max_val : range for scalar grounding (ignored for non-scalars)
        """
        if isinstance(data, dict):
            return self.ground_dict(data, domain)
        elif isinstance(data, (list, tuple)):
            return self.ground_sequence(data, domain, min_val, max_val)
        elif isinstance(data, str):
            return self.ground_category(data, domain)
        elif isinstance(data, (int, float, np.integer, np.floating)):
            return self.ground_scalar(float(data), min_val, max_val, domain)
        else:
            # Fallback: hash repr as category
            return self.ground_category(repr(data), domain)

    # ---- Scalar -------------------------------------------------------
    def ground_scalar(
        self,
        value: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        domain: str = "default",
    ) -> hv.HyperVector:
        """Thermometer encode *value* ∈ [min_val, max_val] into an HV.

        Nearby values produce highly similar vectors (Scalar Similarity Test).
        """
        bins = self._get_bin_hvs(domain)

        # Clamp + normalise to [0, 1]
        span = max_val - min_val
        if span == 0:
            t = 0.5
        else:
            t = (value - min_val) / span
        t = max(0.0, min(1.0, t))

        # Thermometer: activate bins 0 .. k  (k = floor(t * n_bins))
        k = int(t * (self.n_bins - 1))

        # Weighted bundle of the active bins (closer bins get full weight,
        # further bins taper linearly).
        # For efficiency we XOR-bundle neighbours in a window ±3 around k.
        window = 3
        lo = max(0, k - window)
        hi = min(self.n_bins - 1, k + window)

        result = bins[k]  # centre bin gets full weight
        for i in range(lo, hi + 1):
            if i == k:
                continue
            result = result.bundle(bins[i])

        self._stats["scalars_grounded"] += 1
        return result

    # ---- Category -----------------------------------------------------
    def ground_category(self, label: str, domain: str = "default") -> hv.HyperVector:
        """Deterministic HV for a categorical label (with LRU eviction)."""
        key = f"{domain}:{label}"

        if key in self._codebook:
            # Move to end (most recent)
            self._codebook.move_to_end(key)
            self._stats["categories_grounded"] += 1
            return self._codebook[key]

        # Generate new HV from deterministic seed
        seed = _stable_seed(key)
        vec = hv.HyperVector(seed)

        # LRU eviction
        if len(self._codebook) >= self.max_codebook:
            self._codebook.popitem(last=False)

        self._codebook[key] = vec
        self._stats["categories_grounded"] += 1
        self._stats["codebook_size"] = len(self._codebook)
        return vec

    # ---- Dict (recursive role-filler binding) -------------------------
    def ground_dict(self, data: dict, domain: str = "default") -> hv.HyperVector:
        """Bind every (key, value) pair with role-HV, then XOR-bundle."""
        if not data:
            return hv.HyperVector(0)

        parts = []
        for key, val in data.items():
            role_hv = self._get_role_hv(key, domain)
            # Recursively ground the value
            val_hv = self.ground(val, domain=f"{domain}.{key}")
            # Bind role ⊗ filler
            bound = role_hv.xor(val_hv)
            parts.append(bound)

        # XOR-bundle all pairs
        result = parts[0]
        for p in parts[1:]:
            result = result.xor(p)

        self._stats["dicts_grounded"] += 1
        return result

    # ---- List (sequence encoding) -------------------------------------
    def ground_sequence(
        self,
        data: list,
        domain: str = "default",
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> hv.HyperVector:
        """Encode an ordered sequence using permutation: Σ ρ^i(item_i).

        Preserves order: [A, B, C] ≠ [B, A, C].
        """
        if not data:
            return hv.HyperVector(0)

        parts = []
        for i, item in enumerate(data):
            item_hv = self.ground(item, domain=domain, min_val=min_val, max_val=max_val)
            # Apply i-th permutation for positional encoding
            if i > 0:
                item_hv = item_hv.permute(i)
            parts.append(item_hv)

        # XOR-bundle all position-encoded items
        result = parts[0]
        for p in parts[1:]:
            result = result.xor(p)

        self._stats["lists_grounded"] += 1
        return result

    # ------------------------------------------------------------------
    # Introspection (for dashboard telemetry)
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return grounding statistics for dashboard display."""
        self._stats["codebook_size"] = len(self._codebook)
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_bin_hvs(self, domain: str) -> list:
        """Lazily generate n_bins base HVs for thermometer encoding."""
        if domain not in self._bin_hvs:
            base_seed = _stable_seed(f"__thermo__{domain}")
            bins = []
            for i in range(self.n_bins):
                bins.append(hv.HyperVector((base_seed + i) & 0xFFFF_FFFF))
            self._bin_hvs[domain] = bins
        return self._bin_hvs[domain]

    def _get_role_hv(self, key: str, domain: str) -> hv.HyperVector:
        """Lazily generate a role HV for a dict key."""
        full_key = f"__role__{domain}:{key}"
        if full_key not in self._role_hvs:
            self._role_hvs[full_key] = hv.HyperVector(_stable_seed(full_key))
        return self._role_hvs[full_key]
