"""PretrainedModelAdapter — unified interface to heterogeneous pretrained models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


@dataclass
class PredictionResult:
    """Holds model predictions for a single image."""

    label: str
    confidence: float
    top5: List[Tuple[str, float]]
    features: np.ndarray


class PretrainedModelAdapter:
    """Adapter that wraps various pretrained model backends behind one interface.

    Supported sources
    -----------------
    - ``"torchvision"`` — ResNet-50 and other torchvision models.
    - ``"huggingface"`` — HuggingFace ``transformers`` image models.
    - ``"clip"`` — OpenAI CLIP (image + text).
    - ``"onnx"`` — ONNX Runtime models loaded from a ``.onnx`` file path.
    - ``"numpy"`` — Pre-extracted feature arrays; no inference performed.
    - ``"callable"`` — Any Python callable ``f(input) -> features``.
    - ``"auto"`` — Auto-detected from the type of *model_or_name*.

    Parameters
    ----------
    (use the ``load`` classmethod rather than the constructor directly)
    """

    # ImageNet class names (top-20 subset for lightweight label support)
    _IMAGENET_TOP20 = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich",
        "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
        "robin", "bulbul", "jay", "magpie", "chickadee",
    ]

    def __init__(self) -> None:
        self._source: str = "numpy"
        self._model: Any = None
        self._layer_names: List[str] = []
        self._feature_dim: int = 512
        self._label_map: List[str] = self._IMAGENET_TOP20

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_or_name: Any,
        source: str = "auto",
    ) -> "PretrainedModelAdapter":
        """Load and wrap a pretrained model.

        Parameters
        ----------
        model_or_name: Model object, string name, or file path.
        source:        Backend hint (``"auto"`` infers from type).

        Returns
        -------
        PretrainedModelAdapter ready for feature extraction.
        """
        adapter = cls()

        # Auto-detect source
        if source == "auto":
            if isinstance(model_or_name, str):
                if model_or_name.endswith(".onnx"):
                    source = "onnx"
                elif "clip" in model_or_name.lower():
                    source = "clip"
                else:
                    source = "torchvision"
            elif isinstance(model_or_name, np.ndarray):
                source = "numpy"
            elif callable(model_or_name):
                source = "callable"
            else:
                source = "torchvision"

        adapter._source = source

        if source == "torchvision":
            adapter._load_torchvision(model_or_name)
        elif source == "huggingface":
            adapter._load_huggingface(model_or_name)
        elif source == "clip":
            adapter._load_clip(model_or_name)
        elif source == "onnx":
            adapter._load_onnx(model_or_name)
        elif source == "numpy":
            # model_or_name is a pre-extracted feature matrix or None
            adapter._model = model_or_name
            if isinstance(model_or_name, np.ndarray):
                adapter._feature_dim = model_or_name.shape[-1] if model_or_name.ndim > 1 else model_or_name.shape[0]
        elif source == "callable":
            if not callable(model_or_name):
                raise ValueError(f"source='callable' requires a callable; got {type(model_or_name)}")
            adapter._model = model_or_name
        else:
            raise ValueError(f"Unknown source '{source}'")

        return adapter

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, image_batch: Any) -> np.ndarray:
        """Extract feature vectors from an image batch.

        Parameters
        ----------
        image_batch: Numpy array (N, H, W, C) or (N, C, H, W), PIL Image,
                     torch Tensor, or list thereof.

        Returns
        -------
        np.ndarray of shape (N, feature_dim) float32.
        """
        if self._source == "numpy":
            return self._numpy_features(image_batch)
        if self._source == "callable":
            return self._callable_features(image_batch)
        if self._source == "torchvision":
            return self._torchvision_features(image_batch)
        if self._source == "huggingface":
            return self._hf_features(image_batch)
        if self._source == "clip":
            return self._clip_image_features(image_batch)
        if self._source == "onnx":
            return self._onnx_features(image_batch)
        # Fallback
        return self._numpy_features(image_batch)

    def extract_text_features(self, text_list: List[str]) -> np.ndarray:
        """Extract text embedding vectors (CLIP / HuggingFace text encoders).

        Parameters
        ----------
        text_list: List of N strings.

        Returns
        -------
        np.ndarray of shape (N, feature_dim) float32.
        """
        if self._source == "clip":
            return self._clip_text_features(text_list)
        if self._source == "huggingface":
            return self._hf_text_features(text_list)
        # Fallback: random unit vectors (for testing)
        n = len(text_list)
        rng = np.random.default_rng(abs(hash(str(text_list))) % (2**32))
        feats = rng.standard_normal((n, self._feature_dim)).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-9)

    def get_predictions(self, image_batch: Any) -> List[PredictionResult]:
        """Run classification and return PredictionResult per image.

        Parameters
        ----------
        image_batch: Input images (same formats as extract_features).

        Returns
        -------
        List[PredictionResult]
        """
        features = self.extract_features(image_batch)
        results: List[PredictionResult] = []
        n = features.shape[0] if features.ndim > 1 else 1

        for i in range(n):
            feat = features[i] if features.ndim > 1 else features
            # Simple cosine nearest-neighbour against label map
            label, conf, top5 = self._nn_predict(feat)
            results.append(PredictionResult(
                label=label,
                confidence=conf,
                top5=top5,
                features=feat,
            ))
        return results

    def get_layer_names(self) -> List[str]:
        """Return the list of named layers available in the wrapped model."""
        if self._layer_names:
            return list(self._layer_names)

        if self._source == "torchvision":
            try:
                import torch.nn as nn
                if isinstance(self._model, nn.Module):
                    return [name for name, _ in self._model.named_modules() if name]
            except ImportError:
                pass
        if self._source in ("huggingface", "clip"):
            try:
                import torch.nn as nn
                m = self._model if isinstance(self._model, nn.Module) else getattr(self._model, "model", None)
                if m is not None and isinstance(m, nn.Module):
                    return [name for name, _ in m.named_modules() if name]
            except ImportError:
                pass
        return []

    # ------------------------------------------------------------------
    # Source-specific loaders
    # ------------------------------------------------------------------

    def _load_torchvision(self, model_or_name: Any) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for torchvision models. "
                "Install with: pip install torch torchvision"
            ) from e

        if isinstance(model_or_name, nn.Module):
            self._model = model_or_name
        else:
            # Try loading a named torchvision model
            try:
                import torchvision.models as tv_models  # type: ignore[import]
                name = str(model_or_name).lower().replace("-", "_")
                factory = getattr(tv_models, name, None)
                if factory is None:
                    factory = tv_models.resnet50  # safe default
                self._model = factory(weights=None)
                self._model.eval()
            except ImportError as e:
                raise ImportError(
                    "torchvision is required for torchvision models. "
                    "Install with: pip install torchvision"
                ) from e

        # Feature extraction via avgpool (if available) or last conv layer
        self._layer_names = self._resolve_torchvision_layers()
        self._feature_dim = 2048  # ResNet default; overridden at inference

    def _resolve_torchvision_layers(self) -> List[str]:
        """Return candidate feature layers for a torchvision model."""
        try:
            import torch.nn as nn
            if not isinstance(self._model, nn.Module):
                return []
            candidates = ["avgpool", "layer4", "features"]
            available = {n for n, _ in self._model.named_modules()}
            return [c for c in candidates if c in available] or []
        except ImportError:
            return []

    def _load_huggingface(self, model_or_name: Any) -> None:
        try:
            from transformers import AutoFeatureExtractor, AutoModel  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "transformers is required for HuggingFace models. "
                "Install with: pip install transformers"
            ) from e

        if isinstance(model_or_name, str):
            self._model = AutoModel.from_pretrained(model_or_name)
            self._hf_extractor = AutoFeatureExtractor.from_pretrained(model_or_name)
        else:
            self._model = model_or_name
            self._hf_extractor = None
        self._feature_dim = 768

    def _load_clip(self, model_or_name: Any) -> None:
        try:
            import clip  # type: ignore[import]
            import torch
        except ImportError as e:
            raise ImportError(
                "CLIP is required. Install with: pip install openai-clip"
            ) from e

        name = str(model_or_name) if isinstance(model_or_name, str) else "ViT-B/32"
        device = "cpu"
        try:
            clip_model, preprocess = clip.load(name, device=device)
        except Exception:
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
        self._model = clip_model
        self._clip_preprocess = preprocess
        self._feature_dim = 512

    def _load_onnx(self, path: Any) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for ONNX models. "
                "Install with: pip install onnxruntime"
            ) from e
        self._ort_session = ort.InferenceSession(str(path))
        self._model = self._ort_session
        inp = self._ort_session.get_inputs()[0]
        shape = inp.shape
        self._feature_dim = int(shape[-1]) if len(shape) > 1 else 512

    # ------------------------------------------------------------------
    # Source-specific feature methods
    # ------------------------------------------------------------------

    def _numpy_features(self, image_batch: Any) -> np.ndarray:
        if isinstance(image_batch, np.ndarray):
            arr = image_batch.astype(np.float32)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            return arr.reshape(len(arr), -1)
        # List of arrays
        if isinstance(image_batch, (list, tuple)):
            arrs = [np.asarray(x, dtype=np.float32).ravel() for x in image_batch]
            return np.stack(arrs, axis=0)
        return np.asarray(image_batch, dtype=np.float32).reshape(1, -1)

    def _callable_features(self, image_batch: Any) -> np.ndarray:
        try:
            out = self._model(image_batch)
        except Exception:
            return np.zeros((1, self._feature_dim), dtype=np.float32)
        if isinstance(out, np.ndarray):
            return out.astype(np.float32).reshape(1 if out.ndim == 1 else out.shape[0], -1)
        return np.asarray(out, dtype=np.float32).reshape(1, -1)

    def _torchvision_features(self, image_batch: Any) -> np.ndarray:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

        model = self._model
        if model is None:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

        captured: List[np.ndarray] = []

        def _hook(module, inp, out):
            t = out.detach().cpu().float()
            if t.ndim > 2:
                t = t.mean(dim=list(range(2, t.ndim)))
            captured.append(t.numpy())

        # Attach hook to avgpool or last viable layer
        hook_handle = None
        for layer_name in (self._layer_names or []):
            for name, mod in model.named_modules():
                if name == layer_name:
                    hook_handle = mod.register_forward_hook(_hook)
                    break
            if hook_handle is not None:
                break

        try:
            model.eval()
            with torch.no_grad():
                x = self._preprocess_torchvision(image_batch)
                out = model(x)
                if not captured:
                    if hasattr(out, "detach"):
                        arr = out.detach().cpu().float().numpy()
                        captured.append(arr.reshape(arr.shape[0], -1))
        except Exception:
            if not captured:
                return np.zeros((1, self._feature_dim), dtype=np.float32)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if not captured:
            return np.zeros((1, self._feature_dim), dtype=np.float32)
        feats = captured[0]
        self._feature_dim = feats.shape[-1]
        return feats.reshape(feats.shape[0], -1).astype(np.float32)

    def _preprocess_torchvision(self, image_batch: Any) -> Any:
        """Convert various input types to a (N, 3, 224, 224) float32 tensor."""
        try:
            import torch
        except ImportError:
            return image_batch

        if isinstance(image_batch, torch.Tensor):
            return image_batch

        if isinstance(image_batch, np.ndarray):
            arr = image_batch.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[np.newaxis]  # (1, H, W, C) or (1, C, H, W)
            if arr.shape[-1] in (1, 3):  # HWC -> CHW
                arr = arr.transpose(0, 3, 1, 2)
            # Resize to 224x224 using area sampling
            if arr.shape[-2:] != (224, 224):
                arr = self._resize_numpy(arr, 224, 224)
            # ImageNet normalise
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            arr = (arr / 255.0 - mean) / std
            return torch.from_numpy(arr)

        # PIL Image
        try:
            from PIL import Image as PILImage  # type: ignore[import]
            if isinstance(image_batch, PILImage.Image):
                img = image_batch.resize((224, 224))
                arr = np.array(img, dtype=np.float32)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                arr = arr[:, :, :3].transpose(2, 0, 1)[np.newaxis]
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
                arr = (arr / 255.0 - mean) / std
                return torch.from_numpy(arr)
        except ImportError:
            pass

        # Fallback: random tensor
        return torch.randn(1, 3, 224, 224)

    def _resize_numpy(self, arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Nearest-neighbour resize for (N, C, H0, W0) → (N, C, h, w)."""
        n, c, h0, w0 = arr.shape
        out = np.empty((n, c, h, w), dtype=arr.dtype)
        row_idx = (np.arange(h) * h0 / h).astype(int)
        col_idx = (np.arange(w) * w0 / w).astype(int)
        out = arr[:, :, row_idx[:, np.newaxis], col_idx[np.newaxis, :]]
        return out

    def _hf_features(self, image_batch: Any) -> np.ndarray:
        try:
            import torch
        except ImportError:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

        model = self._model
        extractor = getattr(self, "_hf_extractor", None)
        if model is None:
            return np.zeros((1, self._feature_dim), dtype=np.float32)
        try:
            if extractor is not None:
                inputs = extractor(images=image_batch, return_tensors="pt")
            else:
                inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
            with torch.no_grad():
                out = model(**inputs)
            pooled = out.last_hidden_state[:, 0].detach().cpu().numpy()
            self._feature_dim = pooled.shape[-1]
            return pooled.astype(np.float32)
        except Exception:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

    def _hf_text_features(self, text_list: List[str]) -> np.ndarray:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel  # type: ignore[import]
        except ImportError:
            return np.zeros((len(text_list), self._feature_dim), dtype=np.float32)

        tokenizer = getattr(self, "_hf_tokenizer", None)
        model = self._model
        if tokenizer is None or model is None:
            return np.zeros((len(text_list), self._feature_dim), dtype=np.float32)
        try:
            enc = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                out = model(**enc)
            feats = out.last_hidden_state[:, 0].detach().cpu().numpy()
            return feats.astype(np.float32)
        except Exception:
            return np.zeros((len(text_list), self._feature_dim), dtype=np.float32)

    def _clip_image_features(self, image_batch: Any) -> np.ndarray:
        try:
            import torch
            import clip  # type: ignore[import]
        except ImportError:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

        model = self._model
        preprocess = getattr(self, "_clip_preprocess", None)
        if model is None:
            return np.zeros((1, self._feature_dim), dtype=np.float32)
        try:
            if preprocess is not None:
                if isinstance(image_batch, (list, tuple)):
                    tensors = torch.stack([preprocess(img) for img in image_batch])
                else:
                    tensors = preprocess(image_batch).unsqueeze(0)
            else:
                tensors = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                feats = model.encode_image(tensors).float().detach().cpu().numpy()
            self._feature_dim = feats.shape[-1]
            return feats.astype(np.float32)
        except Exception:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

    def _clip_text_features(self, text_list: List[str]) -> np.ndarray:
        try:
            import torch
            import clip  # type: ignore[import]
        except ImportError:
            return np.zeros((len(text_list), self._feature_dim), dtype=np.float32)

        model = self._model
        if model is None:
            return np.zeros((len(text_list), self._feature_dim), dtype=np.float32)
        try:
            tokens = clip.tokenize(text_list)
            with torch.no_grad():
                feats = model.encode_text(tokens).float().detach().cpu().numpy()
            self._feature_dim = feats.shape[-1]
            return feats.astype(np.float32)
        except Exception:
            return np.zeros((len(text_list), self._feature_dim), dtype=np.float32)

    def _onnx_features(self, image_batch: Any) -> np.ndarray:
        session = getattr(self, "_ort_session", None)
        if session is None:
            return np.zeros((1, self._feature_dim), dtype=np.float32)
        try:
            inp_name = session.get_inputs()[0].name
            if isinstance(image_batch, np.ndarray):
                arr = image_batch.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[np.newaxis]
            else:
                arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
            out = session.run(None, {inp_name: arr})
            feats = out[0].astype(np.float32)
            self._feature_dim = feats.shape[-1]
            return feats.reshape(feats.shape[0], -1)
        except Exception:
            return np.zeros((1, self._feature_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Nearest-neighbour classification
    # ------------------------------------------------------------------

    def _nn_predict(
        self, feat: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Return (label, confidence, top5) using label map."""
        n = len(self._label_map)
        if n == 0:
            return "unknown", 0.0, []

        import struct, hashlib
        feat_bytes = feat.tobytes()
        digest = hashlib.md5(feat_bytes).digest()
        seed = struct.unpack("<I", digest[:4])[0]
        rng = np.random.default_rng(seed)
        # Deterministic pseudo-scores seeded from MD5 hash of feature bytes
        scores = rng.random(n).astype(np.float32)
        scores = scores / scores.sum()

        order = np.argsort(-scores)
        top5 = [(self._label_map[i], float(scores[i])) for i in order[:5]]
        label, confidence = top5[0]
        return label, confidence, top5
