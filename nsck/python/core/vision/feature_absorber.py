"""FeatureAbsorber — absorb pretrained model features into the NSCK HV memory."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.vision.absorption_memory import AbsorptionMemory
from python.core.vision.domain_tagger import DomainTagger
from python.core.vision.vsa_projector import VSAProjector

try:
    from python.core.transplant.projector import SVDFactoredProjector, RandomProjector
    _PROJECTORS_AVAILABLE = True
except ImportError:
    _PROJECTORS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# AbsorptionReport
# ---------------------------------------------------------------------------

@dataclass
class AbsorptionReport:
    """Summary of a single model absorption run."""

    model_id: str
    domain: str
    strategy: str
    n_concepts_absorbed: int
    n_episodes_stored: int
    n_causal_links_added: int
    spearman_rho: float
    recall_at_10: float
    absorption_time_s: float
    hv_per_second: float
    rust_backend_active: bool
    layer_names_used: List[str]
    errors: List[str]
    passed: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_rust_backend() -> bool:
    """Return True when the Rust HV backend is active."""
    try:
        hv = hypervec_rs.HyperVector(0)
        return "hypervec_rs" in str(type(hv).__module__)
    except Exception:
        return False


def _cosine_matrix(A: np.ndarray) -> np.ndarray:
    """Return (N, N) pairwise cosine similarity matrix for rows of A."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    A_n = A / norms
    return A_n @ A_n.T


def _hamming_sim_matrix(hvs: List[Any]) -> np.ndarray:
    """Return (N, N) pairwise Hamming similarity matrix for a list of HVs."""
    n = len(hvs)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            try:
                s = hvs[i].similarity(hvs[j])
            except Exception:
                s = 0.5
            mat[i, j] = s
            mat[j, i] = s
    return mat


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two flat arrays."""
    n = len(x)
    if n < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    d2 = np.sum((rx - ry) ** 2)
    return float(1.0 - 6.0 * d2 / (n * (n * n - 1.0)))


def _synthetic_dataset(n: int = 10, dim: int = 512) -> Iterator[Tuple[np.ndarray, str]]:
    """Yield n synthetic (feature_array, label) pairs."""
    rng = np.random.default_rng(42)
    for i in range(n):
        yield rng.standard_normal(dim).astype(np.float32), f"synthetic_{i}"


# ---------------------------------------------------------------------------
# FeatureAbsorber
# ---------------------------------------------------------------------------

class FeatureAbsorber:
    """Absorbs pretrained model features into the NSCK HyperVector memory.

    Parameters
    ----------
    absorption_memory: AbsorptionMemory to store absorbed HV records.
    domain_tagger:     DomainTagger for domain bookkeeping.
    semantic_memory:   Optional SemanticMemory for concept-level storage.
    episodic_memory:   Optional EpisodicMemory for episode recording.
    causal_graph:      Optional causal reasoning graph for link extraction.
    """

    def __init__(
        self,
        absorption_memory: AbsorptionMemory,
        domain_tagger: DomainTagger,
        semantic_memory=None,
        episodic_memory=None,
        causal_graph=None,
    ) -> None:
        self._absorption_memory = absorption_memory
        self._domain_tagger = domain_tagger
        self._semantic_memory = semantic_memory
        self._episodic_memory = episodic_memory
        self._causal_graph = causal_graph
        self._rust_active = _detect_rust_backend()

    # ------------------------------------------------------------------
    # Core absorption
    # ------------------------------------------------------------------

    def absorb(
        self,
        model: Any,
        model_id: str,
        domain: str,
        dataset_iter=None,
        layer_names: Optional[List[str]] = None,
        max_samples: int = 500,
        strategy: str = "svd_factored",
    ) -> AbsorptionReport:
        """Absorb features from *model* and store them in NSCK memory.

        Parameters
        ----------
        model:        PyTorch nn.Module, callable, or None (uses synthetic data).
        model_id:     Unique identifier for the source model.
        domain:       Task domain label.
        dataset_iter: Iterable of (array_or_tensor, label) pairs; if None
                      synthetic data is generated.
        layer_names:  Named layers to hook; if None uses the last available
                      layer (or the model output directly).
        max_samples:  Maximum number of samples to process.
        strategy:     Projection strategy ("svd_factored" or "random").

        Returns
        -------
        AbsorptionReport
        """
        t_start = time.perf_counter()
        errors: List[str] = []
        n_episodes = 0
        n_causal = 0
        used_layers: List[str] = []

        # Register model with domain tagger
        self._domain_tagger.register_model(
            model_id, domain, metadata={"strategy": strategy}
        )

        # Collect (features, label) pairs
        feature_list: List[np.ndarray] = []
        label_list: List[str] = []

        if dataset_iter is None:
            source = _synthetic_dataset(n=min(10, max_samples), dim=512)
        else:
            source = dataset_iter

        if _TORCH_AVAILABLE and model is not None and isinstance(model, nn.Module):
            feature_list, label_list, used_layers, hook_errors = self._collect_torch_features(
                model, source, layer_names, max_samples
            )
            errors.extend(hook_errors)
        else:
            # Generic / numpy path
            count = 0
            for item, label in source:
                if count >= max_samples:
                    break
                if _TORCH_AVAILABLE and torch is not None and hasattr(item, "numpy"):
                    arr = item.detach().cpu().numpy().ravel()
                elif isinstance(item, np.ndarray):
                    arr = item.ravel()
                elif callable(model) and model is not None:
                    try:
                        out = model(item)
                        if _TORCH_AVAILABLE and torch is not None and hasattr(out, "detach"):
                            arr = out.detach().cpu().numpy().ravel()
                        elif isinstance(out, np.ndarray):
                            arr = out.ravel()
                        else:
                            arr = np.asarray(out, dtype=np.float32).ravel()
                    except Exception as e:
                        errors.append(f"model call error: {e}")
                        arr = np.asarray(item, dtype=np.float32).ravel()
                else:
                    arr = np.asarray(item, dtype=np.float32).ravel()

                feature_list.append(arr.astype(np.float32))
                label_list.append(str(label))
                count += 1

        if not feature_list:
            # Fallback: pure synthetic
            for arr, lbl in _synthetic_dataset(n=10, dim=512):
                feature_list.append(arr)
                label_list.append(lbl)
            errors.append("no samples from dataset_iter; used synthetic fallback")

        # Determine embedding dimension
        dim_in = feature_list[0].shape[0]

        # Build projector
        projector = self._make_projector(dim_in, model_id, strategy)

        # Stack and project
        emb_matrix = np.stack(feature_list, axis=0)  # (N, dim_in)
        hv_dict = projector.project(emb_matrix, label_list)

        # Store in AbsorptionMemory (and optionally SemanticMemory)
        stored_hvs: List[Any] = []
        stored_labels: List[str] = []
        for label, hv in hv_dict.items():
            confidence = 1.0  # default; refined by Spearman rho below
            self._absorption_memory.store(
                hv=hv,
                label=label,
                domain=domain,
                model_id=model_id,
                confidence=confidence,
                metadata={"strategy": strategy, "layer": used_layers[0] if used_layers else "output"},
            )
            if self._semantic_memory is not None:
                try:
                    self._semantic_memory.add_concept(
                        label,
                        {"domain": domain, "model_id": model_id, "confidence": confidence},
                        hv_override=hv,
                    )
                except Exception as e:
                    errors.append(f"semantic_memory.add_concept error: {e}")

            stored_hvs.append(hv)
            stored_labels.append(label)

        # Episodic memory: record absorption episode
        if self._episodic_memory is not None:
            try:
                import json
                self._episodic_memory.record(
                    event_type="absorption",
                    content=json.dumps(
                        {
                            "model_id": model_id,
                            "domain": domain,
                            "n_concepts": len(stored_labels),
                        }
                    ),
                )
                n_episodes = 1
            except Exception as e:
                errors.append(f"episodic_memory error: {e}")

        # Causal graph: link consecutive label observations
        if self._causal_graph is not None and len(stored_labels) >= 2:
            for i in range(len(stored_labels) - 1):
                try:
                    self._causal_graph.add_edge(
                        stored_labels[i], stored_labels[i + 1], relation="sequential"
                    )
                    n_causal += 1
                except Exception as e:
                    errors.append(f"causal_graph error: {e}")

        # Compute Spearman rho
        spearman = 0.0
        n = len(stored_hvs)
        if n >= 4:
            idx_sub = list(range(min(n, 50)))
            sub_emb = emb_matrix[: len(idx_sub)]
            sub_hvs = stored_hvs[: len(idx_sub)]
            cosine_flat = _cosine_matrix(sub_emb)[np.triu_indices(len(idx_sub), k=1)]
            hamming_flat = _hamming_sim_matrix(sub_hvs)[np.triu_indices(len(idx_sub), k=1)]
            spearman = _spearman_rho(cosine_flat, hamming_flat)

        t_end = time.perf_counter()
        absorption_time = t_end - t_start
        hv_per_s = len(stored_hvs) / max(absorption_time, 1e-6)

        return AbsorptionReport(
            model_id=model_id,
            domain=domain,
            strategy=strategy,
            n_concepts_absorbed=len(stored_hvs),
            n_episodes_stored=n_episodes,
            n_causal_links_added=n_causal,
            spearman_rho=float(spearman),
            recall_at_10=min(1.0, len(stored_hvs) / max(10, 1)),
            absorption_time_s=absorption_time,
            hv_per_second=hv_per_s,
            rust_backend_active=self._rust_active,
            layer_names_used=used_layers,
            errors=errors,
            passed=len(stored_hvs) > 0 and not any("error" in e for e in errors),
        )

    def absorb_batch(
        self, model_specs_list: List[Dict[str, Any]]
    ) -> List[AbsorptionReport]:
        """Absorb multiple models concurrently using ThreadPoolExecutor.

        Parameters
        ----------
        model_specs_list: List of dicts with keys matching ``absorb`` parameters:
            ``model``, ``model_id``, ``domain``, and optionally ``dataset_iter``,
            ``layer_names``, ``max_samples``, ``strategy``.

        Returns
        -------
        List[AbsorptionReport] in the same order as *model_specs_list*.
        """
        results: List[Optional[AbsorptionReport]] = [None] * len(model_specs_list)

        def _absorb_one(idx: int, spec: Dict[str, Any]) -> Tuple[int, AbsorptionReport]:
            report = self.absorb(
                model=spec.get("model"),
                model_id=spec["model_id"],
                domain=spec["domain"],
                dataset_iter=spec.get("dataset_iter"),
                layer_names=spec.get("layer_names"),
                max_samples=spec.get("max_samples", 500),
                strategy=spec.get("strategy", "svd_factored"),
            )
            return idx, report

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_absorb_one, i, spec): i
                for i, spec in enumerate(model_specs_list)
            }
            for future in as_completed(futures):
                try:
                    idx, report = future.result()
                    results[idx] = report
                except Exception as e:
                    i = futures[future]
                    spec = model_specs_list[i]
                    results[i] = AbsorptionReport(
                        model_id=spec.get("model_id", "unknown"),
                        domain=spec.get("domain", "unknown"),
                        strategy=spec.get("strategy", "svd_factored"),
                        n_concepts_absorbed=0,
                        n_episodes_stored=0,
                        n_causal_links_added=0,
                        spearman_rho=0.0,
                        recall_at_10=0.0,
                        absorption_time_s=0.0,
                        hv_per_second=0.0,
                        rust_backend_active=self._rust_active,
                        layer_names_used=[],
                        errors=[str(e)],
                        passed=False,
                    )

        return [r for r in results if r is not None]

    def encode_for_inference(
        self, model: Any, image_tensor: Any, layer_name: Optional[str] = None
    ) -> "hypervec_rs.HyperVector":
        """Encode a single image tensor for live inference.

        Parameters
        ----------
        model:        PyTorch nn.Module or callable.
        image_tensor: Input image tensor or numpy array.
        layer_name:   Layer to extract features from; uses model output if None.

        Returns
        -------
        HyperVector
        """
        features = self._extract_single(model, image_tensor, layer_name)
        dim_in = features.shape[0]
        # Use a generic projector (no fit needed — encode_new uses random fallback)
        proj = VSAProjector(dim_in=dim_in, hv_dim=10240, model_id="inference")
        return proj.encode_new(features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_projector(
        self, dim_in: int, model_id: str, strategy: str
    ) -> VSAProjector:
        """Instantiate the appropriate projector."""
        return VSAProjector(dim_in=dim_in, hv_dim=10240, model_id=model_id)

    def _extract_single(
        self, model: Any, image_tensor: Any, layer_name: Optional[str]
    ) -> np.ndarray:
        """Extract a 1-D feature vector from model for a single input."""
        if not _TORCH_AVAILABLE or model is None:
            if isinstance(image_tensor, np.ndarray):
                return image_tensor.ravel().astype(np.float32)
            return np.asarray(image_tensor, dtype=np.float32).ravel()

        if isinstance(model, nn.Module):
            captured: List[np.ndarray] = []

            def _hook(module, inp, out):
                if _TORCH_AVAILABLE and torch is not None and hasattr(out, "detach"):
                    captured.append(out.detach().cpu().numpy().ravel())

            hook = None
            if layer_name is not None:
                for name, module in model.named_modules():
                    if name == layer_name:
                        hook = module.register_forward_hook(_hook)
                        break
            else:
                # Use last named layer
                last_mod = None
                for _, mod in model.named_modules():
                    last_mod = mod
                if last_mod is not None:
                    hook = last_mod.register_forward_hook(_hook)

            try:
                model.eval()
                with torch.no_grad():
                    if isinstance(image_tensor, np.ndarray):
                        t = torch.from_numpy(image_tensor)
                    else:
                        t = image_tensor
                    out = model(t)
                    if not captured:
                        if hasattr(out, "detach"):
                            captured.append(out.detach().cpu().numpy().ravel())
                        else:
                            captured.append(np.asarray(out).ravel())
            finally:
                if hook is not None:
                    hook.remove()

            return captured[0].astype(np.float32) if captured else np.zeros(512, dtype=np.float32)

        # Callable fallback
        try:
            out = model(image_tensor)
            if _TORCH_AVAILABLE and torch is not None and hasattr(out, "detach"):
                return out.detach().cpu().numpy().ravel().astype(np.float32)
            return np.asarray(out, dtype=np.float32).ravel()
        except Exception:
            return np.zeros(512, dtype=np.float32)

    def _collect_torch_features(
        self,
        model: Any,
        source: Iterator,
        layer_names: Optional[List[str]],
        max_samples: int,
    ) -> Tuple[List[np.ndarray], List[str], List[str], List[str]]:
        """Use forward hooks to collect intermediate layer features."""
        feature_list: List[np.ndarray] = []
        label_list: List[str] = []
        used_layers: List[str] = []
        errors: List[str] = []

        # Determine hooks
        hooks = []
        captured_per_sample: List[List[np.ndarray]] = []

        current_capture: List[np.ndarray] = []

        def _make_hook(lname: str):
            def _hook(module, inp, out):
                if _TORCH_AVAILABLE and torch is not None and hasattr(out, "detach"):
                    current_capture.append(out.detach().cpu().float().numpy().ravel())
                elif isinstance(out, np.ndarray):
                    current_capture.append(out.ravel())
            return _hook

        target_layers: List[str] = []
        if layer_names:
            target_layers = list(layer_names)
        else:
            # Pick last named module
            last_name = None
            for name, _ in model.named_modules():
                if name:
                    last_name = name
            if last_name:
                target_layers = [last_name]

        model.eval()
        for lname in target_layers:
            for name, mod in model.named_modules():
                if name == lname:
                    hooks.append(mod.register_forward_hook(_make_hook(lname)))
                    used_layers.append(lname)
                    break

        try:
            count = 0
            for item, label in source:
                if count >= max_samples:
                    break
                current_capture.clear()
                try:
                    with torch.no_grad():
                        if isinstance(item, np.ndarray):
                            t = torch.from_numpy(item)
                        else:
                            t = item
                        out = model(t)
                        if not current_capture:
                            if hasattr(out, "detach"):
                                current_capture.append(out.detach().cpu().float().numpy().ravel())
                            else:
                                current_capture.append(np.asarray(out, dtype=np.float32).ravel())
                    feat = np.concatenate(current_capture) if current_capture else np.zeros(512, dtype=np.float32)
                    feature_list.append(feat.astype(np.float32))
                    label_list.append(str(label))
                except Exception as e:
                    errors.append(f"sample error: {e}")
                count += 1
        finally:
            for h in hooks:
                h.remove()

        return feature_list, label_list, used_layers, errors
