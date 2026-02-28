"""Model Harvester — extracts embedding matrices from external neural models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np


@dataclass
class HarvestResult:
    """Container for harvested embedding data."""
    embeddings: np.ndarray          # shape (vocab_size, embedding_dim)
    vocab_mapping: Dict[str, int]   # token → index
    model_type: str                 # "transformer_lm", "transformer_vision", "cnn",
                                    # "encoder_decoder", "generic", "error"
    embedding_dim: int
    vocab_size: int
    source_model: str               # repr of model
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelHarvester:
    """Harvest embedding matrices from external neural network models."""

    # Transformer LM attribute names to probe, in priority order
    _LM_ATTRS = ("embeddings", "embed_tokens", "wte", "word_embeddings")
    # Vision attribute names
    _VISION_ATTRS = ("patch_embed", "features", "conv1")

    def harvest(self, model: Any, method: str = "auto") -> HarvestResult:
        """Extract embedding matrix from *model*.

        Parameters
        ----------
        model:
            A PyTorch (or duck-typed) model object.
        method:
            One of ``"auto"``, ``"embedding_layer"``, ``"named_params"``,
            ``"forward_hook"``.

        Returns
        -------
        HarvestResult
            On any failure the ``model_type`` field is ``"error"``.
        """
        try:
            if method == "auto":
                return self._auto_harvest(model)
            elif method == "embedding_layer":
                return self._harvest_embedding_layer(model)
            elif method == "named_params":
                return self._harvest_named_params(model)
            elif method == "forward_hook":
                return self._harvest_forward_hook(model)
            else:
                raise ValueError(f"Unknown harvest method: {method!r}")
        except Exception as exc:  # noqa: BLE001
            return self._error_result(model, str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _auto_harvest(self, model: Any) -> HarvestResult:
        # 1. Transformer LM
        for attr in self._LM_ATTRS:
            if hasattr(model, attr):
                return self._harvest_embedding_layer(model, model_type="transformer_lm")

        # 2. Vision
        for attr in self._VISION_ATTRS:
            if hasattr(model, attr):
                return self._harvest_vision(model)

        # 3. Encoder-decoder
        if hasattr(model, "encoder") and hasattr(model, "decoder"):
            return self._harvest_encoder_decoder(model)

        # 4. Fallback: named_params scan
        return self._harvest_named_params(model, model_type="generic")

    def _get_weight(self, layer: Any) -> np.ndarray:
        """Extract weight matrix from a layer, converting to numpy float32."""
        w = getattr(layer, "weight", None)
        if w is None:
            raise AttributeError("Layer has no .weight attribute")
        data = getattr(w, "data", w)
        # Support torch tensors (.numpy() / .detach().numpy()) and plain arrays
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "numpy"):
            data = data.numpy()
        return np.asarray(data, dtype=np.float32)

    def _harvest_embedding_layer(
        self, model: Any, model_type: str = "transformer_lm"
    ) -> HarvestResult:
        layer = None
        for attr in self._LM_ATTRS:
            candidate = getattr(model, attr, None)
            if candidate is not None:
                layer = candidate
                break

        if layer is None:
            raise AttributeError("No embedding layer found via LM attrs")

        # Some models nest the embedding: e.g. model.embeddings.word_embeddings
        for sub in ("word_embeddings", "embed_tokens", "wte"):
            sub_layer = getattr(layer, sub, None)
            if sub_layer is not None and hasattr(sub_layer, "weight"):
                layer = sub_layer
                break

        emb = self._get_weight(layer)  # (vocab_size, dim)
        return self._build_result(emb, model, model_type)

    def _harvest_vision(self, model: Any) -> HarvestResult:
        # Strategy 1: Try classifier head first (gives class-level embeddings)
        for clf_attr in ("classifier", "fc", "head", "heads"):
            clf = getattr(model, clf_attr, None)
            if clf is None:
                continue
            # Walk into Sequential classifiers (e.g. MobileNetV2 classifier=[Dropout, Linear])
            if hasattr(clf, '__iter__'):
                for sub in clf:
                    try:
                        w = self._get_weight(sub)
                        if w.ndim == 2 and w.shape[0] > 1:
                            return self._build_result(w, model, "vision_classifier")
                    except (AttributeError, Exception):
                        continue
            else:
                try:
                    w = self._get_weight(clf)
                    if w.ndim == 2 and w.shape[0] > 1:
                        return self._build_result(w, model, "vision_classifier")
                except (AttributeError, Exception):
                    pass

        # Strategy 2: Find patch_embed or first conv layer (original approach)
        for attr in self._VISION_ATTRS:
            layer = getattr(model, attr, None)
            if layer is None:
                continue
            # patch_embed may have a sub-projection
            proj = getattr(layer, "proj", layer)
            try:
                w = self._get_weight(proj)  # (out_ch, in_ch, H, W) or (dim, patch)
                # Reshape to 2D: (out, in*H*W)
                w2d = w.reshape(w.shape[0], -1)
                return self._build_result(w2d, model, "transformer_vision")
            except Exception:  # noqa: BLE001
                pass
            # Walk into Sequential/ModuleList to find largest weight matrix
            best_w = None
            best_n = 0
            for child in self._recursive_modules(layer):
                try:
                    w = self._get_weight(child)
                    n = w.size
                    if n > best_n:
                        best_w = w
                        best_n = n
                except (AttributeError, Exception):
                    continue
            if best_w is not None:
                w2d = best_w.reshape(best_w.shape[0], -1)
                return self._build_result(w2d, model, "transformer_vision")

        # Strategy 3: Fallback to named_params scan
        return self._harvest_named_params(model, model_type="vision_fallback")

    @staticmethod
    def _recursive_modules(container: Any, depth: int = 0) -> list:
        """Walk into Sequential/ModuleList and yield leaf modules."""
        if depth > 5:  # prevent infinite recursion
            return []
        results = []
        children = list(getattr(container, "children", lambda: [])())
        if not children:
            # Leaf module
            results.append(container)
        else:
            for child in children:
                results.extend(ModelHarvester._recursive_modules(child, depth + 1))
        return results

    def _harvest_encoder_decoder(self, model: Any) -> HarvestResult:
        encoder = model.encoder
        for attr in self._LM_ATTRS:
            layer = getattr(encoder, attr, None)
            if layer is not None:
                try:
                    emb = self._get_weight(layer)
                    return self._build_result(emb, model, "encoder_decoder")
                except Exception:  # noqa: BLE001
                    continue
        # Fallback: scan encoder named parameters
        return self._harvest_named_params(encoder, model_type="encoder_decoder")

    def _harvest_named_params(
        self, model: Any, model_type: str = "generic"
    ) -> HarvestResult:
        best: np.ndarray | None = None
        best_n = 0

        if not hasattr(model, "named_parameters"):
            raise AttributeError("Model has no named_parameters()")

        for _name, param in model.named_parameters():
            data = getattr(param, "data", param)
            if hasattr(data, "detach"):
                data = data.detach()
            if hasattr(data, "numpy"):
                data = data.numpy()
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 2:
                n, d = arr.shape
                if n > d and n > best_n:
                    best = arr
                    best_n = n

        if best is None:
            raise ValueError("No suitable 2D weight matrix found in named_parameters")

        return self._build_result(best, model, model_type)

    def _harvest_forward_hook(self, model: Any) -> HarvestResult:
        """Run a forward pass with a hook on the first detectable embedding layer."""
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("forward_hook method requires torch") from exc

        captured: list[np.ndarray] = []

        def _hook(_module: Any, _inp: Any, out: Any) -> None:
            data = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
            captured.append(data.astype(np.float32))

        # Register on first layer that looks like an embedding
        handle = None
        for attr in (*self._LM_ATTRS, *self._VISION_ATTRS):
            layer = getattr(model, attr, None)
            if layer is not None and hasattr(layer, "register_forward_hook"):
                handle = layer.register_forward_hook(_hook)
                break

        if handle is None:
            raise AttributeError("No hookable layer found for forward_hook method")

        try:
            # Build a minimal dummy input
            dummy = torch.zeros(1, 8, dtype=torch.long)
            with torch.no_grad():
                try:
                    model(dummy)
                except Exception as exc:  # noqa: BLE001
                    import warnings  # noqa: PLC0415
                    warnings.warn(
                        f"Forward hook model call raised {type(exc).__name__}; "
                        "hook may still have captured activations.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Forward hook did not capture any activations")

        emb = captured[0]
        if emb.ndim == 3:
            # (batch, seq, dim) → (seq, dim)
            emb = emb[0]
        return self._build_result(emb, model, "generic")

    def _build_result(
        self, emb: np.ndarray, model: Any, model_type: str
    ) -> HarvestResult:
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        vocab_size, embedding_dim = emb.shape

        config = getattr(model, "config", None)
        vocab_mapping: Dict[str, int] = {f"token_{i}": i for i in range(vocab_size)}

        return HarvestResult(
            embeddings=emb,
            vocab_mapping=vocab_mapping,
            model_type=model_type,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            source_model=repr(model),
            metadata={"config": repr(config) if config is not None else None},
        )

    def _error_result(self, model: Any, reason: str) -> HarvestResult:
        dummy = np.zeros((1, 1), dtype=np.float32)
        return HarvestResult(
            embeddings=dummy,
            vocab_mapping={},
            model_type="error",
            embedding_dim=1,
            vocab_size=1,
            source_model=repr(model),
            metadata={"error": reason},
        )
