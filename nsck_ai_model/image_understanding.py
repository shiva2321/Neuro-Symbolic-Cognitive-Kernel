"""
NSCK Image Understanding — VSA-based image context understanding
================================================================

What this does:
    Encodes images into the same hypervector space as text, enabling
    cross-modal retrieval (describe an image, find images from text).

How it works (no neural networks):
    1. Classical CV features are extracted: colour histogram, edge
       histogram, spatial layout.
    2. Each feature is encoded as a hypervector using VSA operations.
    3. The image HV is bundled into the same space as text HVs.
    4. Similarity search finds matching text descriptions.

Why no CNN?
    CNNs require heavy matrix multiplication.  Classical CV features +
    VSA encoding achieves interpretable cross-modal retrieval with O(D)
    operations per comparison.
"""

import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import numpy as np

from nsck_ai_model.ai_engine import HyperVector, DIMENSION

logger = logging.getLogger("nsck_ai.image")

# Try to import PIL for image loading
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class ImageEncoder:
    """Encodes images into hypervectors using classical CV features.

    Features extracted (no neural network):
    1. **Colour histogram** — distribution of pixel intensities per channel.
       Encoded by quantising the histogram into bins and creating an HV
       per bin via deterministic seeding.
    2. **Edge histogram** — gradient magnitudes via Sobel-like finite
       differences.  Encoded similarly.
    3. **Spatial layout** — the image is divided into a grid and each
       cell's mean colour is encoded.

    All features are bundled into a single image HV in the same space
    as text HVs, enabling cross-modal search.
    """

    def __init__(self, grid_size: int = 4, colour_bins: int = 16):
        self.grid_size = grid_size
        self.colour_bins = colour_bins

    def encode_image(self, image) -> HyperVector:
        """Encode a PIL Image or numpy array to a hypervector.

        Parameters
        ----------
        image : PIL.Image or numpy.ndarray
            The image to encode.  If numpy, shape should be (H, W, 3).

        Returns
        -------
        HyperVector
        """
        if _PIL_AVAILABLE and isinstance(image, PILImage.Image):
            arr = np.array(image.convert('RGB')).astype(np.float32)
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
        else:
            # Fallback: random HV
            logger.warning("Cannot process image type %s", type(image))
            return HyperVector()

        if arr.ndim == 2:
            # Grayscale — stack to 3 channels
            arr = np.stack([arr, arr, arr], axis=-1)

        hvs = []
        hvs.append(self._colour_histogram_hv(arr))
        hvs.append(self._edge_histogram_hv(arr))
        hvs.append(self._spatial_layout_hv(arr))
        return HyperVector.bundle(hvs)

    def _colour_histogram_hv(self, arr: np.ndarray) -> HyperVector:
        """Encode the colour histogram as a hypervector."""
        hvs = []
        for ch in range(min(3, arr.shape[-1])):
            channel = arr[:, :, ch].ravel()
            hist, _ = np.histogram(channel, bins=self.colour_bins,
                                   range=(0, 256))
            hist = hist / (hist.sum() + 1e-8)  # normalise
            for i, val in enumerate(hist):
                if val > 0.01:
                    bin_hv = HyperVector.from_seed(f"col:ch{ch}:b{i}")
                    # Weight by frequency — permute by quantised weight
                    weight = int(val * 10)
                    hvs.append(bin_hv.permute(weight))
        return HyperVector.bundle(hvs) if hvs else HyperVector()

    def _edge_histogram_hv(self, arr: np.ndarray) -> HyperVector:
        """Encode edge information using simple finite differences."""
        gray = arr.mean(axis=-1)
        # Horizontal edges (finite difference along x)
        dx = np.abs(np.diff(gray, axis=1))
        # Vertical edges (finite difference along y)
        dy = np.abs(np.diff(gray, axis=0))

        # Quantise edge magnitudes into bins
        hvs = []
        for name, edges in [("dx", dx), ("dy", dy)]:
            hist, _ = np.histogram(edges.ravel(), bins=8, range=(0, 128))
            hist = hist / (hist.sum() + 1e-8)
            for i, val in enumerate(hist):
                if val > 0.01:
                    hvs.append(HyperVector.from_seed(f"edge:{name}:b{i}")
                               .permute(int(val * 10)))
        return HyperVector.bundle(hvs) if hvs else HyperVector()

    def _spatial_layout_hv(self, arr: np.ndarray) -> HyperVector:
        """Encode spatial layout by dividing into a grid."""
        h, w = arr.shape[:2]
        cell_h = max(1, h // self.grid_size)
        cell_w = max(1, w // self.grid_size)
        hvs = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y0 = row * cell_h
                x0 = col * cell_w
                cell = arr[y0:y0 + cell_h, x0:x0 + cell_w]
                mean_colour = cell.mean(axis=(0, 1))
                # Quantise mean colour
                qr = int(mean_colour[0] / 32)
                qg = int(mean_colour[1] / 32) if len(mean_colour) > 1 else 0
                qb = int(mean_colour[2] / 32) if len(mean_colour) > 2 else 0
                cell_hv = HyperVector.from_seed(
                    f"grid:{row},{col}:{qr},{qg},{qb}")
                # Bind with position
                pos_hv = HyperVector.from_seed(f"pos:{row},{col}")
                hvs.append(cell_hv.bind(pos_hv))
        return HyperVector.bundle(hvs) if hvs else HyperVector()


class ImageUnderstanding:
    """Cross-modal image understanding using VSA.

    Connects images and text in the same hypervector space.

    Usage::

        from nsck_ai_model.image_understanding import ImageUnderstanding
        from nsck_ai_model.ai_engine import NSCKAIEngine
        from PIL import Image

        engine = NSCKAIEngine()
        iu = ImageUnderstanding(engine)

        # Teach it about an image
        img = Image.open("cat.jpg")
        iu.learn_image(img, "A cat sitting on a mat")

        # Later, find images matching a description
        results = iu.search_by_text("cat")
    """

    def __init__(self, engine):
        self.engine = engine
        self.image_encoder = ImageEncoder()
        self.image_store: Dict[str, Dict[str, Any]] = {}

    def learn_image(self, image, description: str,
                    image_id: Optional[str] = None) -> Dict[str, Any]:
        """Learn an image with its text description.

        The image is encoded to an HV and stored alongside the text
        description's HV.  This enables cross-modal retrieval.
        """
        if image_id is None:
            image_id = hashlib.md5(
                str(id(image)).encode()).hexdigest()[:8]

        img_hv = self.image_encoder.encode_image(image)
        text_hv = self.engine.encoder.encode_sentence(description)
        # Bundle image and text HVs for joint representation
        joint_hv = HyperVector.bundle([img_hv, text_hv])

        # Store in the engine's knowledge system
        concepts = self.engine.encoder.extract_concepts(description)
        self.engine.knowledge.add_concept(
            name=f"image:{image_id}", hv=joint_hv,
            source_sentence=description)
        self.engine.knowledge.record_episode(
            text=f"[Image: {description}]", hv=joint_hv,
            concepts=concepts)

        # Also train the text encoder on the description
        self.engine.train_on_text(description)

        self.image_store[image_id] = {
            'description': description,
            'image_hv': img_hv,
            'text_hv': text_hv,
            'joint_hv': joint_hv,
            'concepts': concepts,
        }

        return {
            'image_id': image_id,
            'concepts': concepts,
            'description': description,
        }

    def search_by_text(self, query: str,
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Find stored images matching a text query."""
        query_hv = self.engine.encoder.encode_sentence(query)
        results = []
        for img_id, data in self.image_store.items():
            sim = query_hv.similarity(data['joint_hv'])
            if sim > 0.35:
                results.append({
                    'image_id': img_id,
                    'description': data['description'],
                    'similarity': round(sim, 3),
                    'concepts': data['concepts'],
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def search_by_image(self, image,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """Find stored images similar to a query image."""
        query_hv = self.image_encoder.encode_image(image)
        results = []
        for img_id, data in self.image_store.items():
            sim = query_hv.similarity(data['image_hv'])
            if sim > 0.35:
                results.append({
                    'image_id': img_id,
                    'description': data['description'],
                    'similarity': round(sim, 3),
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def describe_image(self, image) -> str:
        """Generate a text description of an image by finding the
        closest stored image and returning its description."""
        results = self.search_by_image(image, top_k=1)
        if results:
            return results[0]['description']
        return "I haven't learned enough about images yet to describe this."

    def get_stats(self) -> Dict[str, Any]:
        return {
            'images_stored': len(self.image_store),
            'image_ids': list(self.image_store.keys())[:10],
        }
