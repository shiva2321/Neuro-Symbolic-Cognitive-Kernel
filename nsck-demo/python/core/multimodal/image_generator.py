"""
NSCK Image Generator
====================
Generates images from text descriptions using VSA/hypervectors (NO neural networks).

Core Approach:
1. Text → LinguaCortex → Concept Hypervector
2. Concept HV → Learned Visual Feature HV (via associative memory)
3. Visual Feature HV → Image pixels (inverse image processing)

Training Process:
- Learn associations between text concepts and visual features
- Build a VSA-based "concept-to-visual" memory
- Use bundling to combine multiple concept features

Generation Process:
- Parse input text into concepts
- Retrieve visual features for each concept
- Bundle features into composite visual representation
- Decode features back to pixel space
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
except ImportError:
    from python.core.vsa.hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()

from python.core.language.lingua_cortex import get_lingua_cortex, SemanticFingerprint
from python.core.memory.semantic_memory import SemanticMemory


@dataclass
class VisualFeatures:
    """Visual features extracted/decoded from hypervectors."""
    hog_features: np.ndarray       # 128 features (gradient orientations)
    color_features: np.ndarray      # 24 features (color histograms)
    edge_density: float             # Edge density [0, 1]
    lbp_features: np.ndarray       # 40 features (texture)
    spatial_features: np.ndarray   # 8 features (spatial stats)
    
    def to_array(self) -> np.ndarray:
        """Flatten all features to a single array."""
        return np.concatenate([
            self.hog_features,
            self.color_features,
            [self.edge_density],
            self.lbp_features,
            self.spatial_features
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'VisualFeatures':
        """Reconstruct features from flattened array."""
        idx = 0
        hog = arr[idx:idx+128]; idx += 128
        color = arr[idx:idx+24]; idx += 24
        edge = float(arr[idx]); idx += 1
        lbp = arr[idx:idx+40]; idx += 40
        spatial = arr[idx:idx+8]; idx += 8
        return cls(hog, color, edge, lbp, spatial)
    
    @classmethod
    def random(cls, seed: int = 42) -> 'VisualFeatures':
        """Generate random features for initialization."""
        rng = np.random.RandomState(seed)
        return cls(
            hog_features=rng.rand(128) * 0.1,  # Low magnitude for HOG
            color_features=rng.rand(24) * 0.2,  # Color histogram
            edge_density=rng.rand() * 0.3,
            lbp_features=rng.rand(40) * 0.2,
            spatial_features=rng.rand(8) * 0.5
        )


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    image_size: Tuple[int, int] = (32, 32)  # Output image size
    is_color: bool = True                     # Generate color or grayscale
    smoothing_factor: float = 0.5             # Smoothing for generated images
    contrast_boost: float = 1.2               # Contrast enhancement


class ConceptVisualMemory:
    """
    Associative memory that maps text concepts to visual features using VSA.
    
    Learning: Bind concept HV with visual feature HV
    Retrieval: Query with concept HV to get visual features
    """
    
    def __init__(self):
        # Concept → Visual Feature HV associations
        self.concept_to_visual: Dict[str, Any] = {}  # concept_name → HV
        
        # Visual feature templates for common concepts
        self.feature_templates: Dict[str, VisualFeatures] = {}
        
        # Role hypervectors for binding
        self.role_concept = hypervec_rs.HyperVector(90001)
        self.role_visual = hypervec_rs.HyperVector(90002)
        
    def learn_association(self, concept: str, visual_features: VisualFeatures, concept_hv: Any):
        """
        Learn an association between a concept and its visual features.
        
        Args:
            concept: The concept name (e.g., "cat", "red", "round")
            visual_features: The visual features to associate
            concept_hv: The hypervector representation of the concept
        """
        # Convert visual features to fingerprint → HV
        feat_array = visual_features.to_array()
        feat_str = "_".join(f"{v:.3f}" for v in feat_array)
        seed = int(hashlib.sha256(feat_str.encode()).hexdigest()[:8], 16)
        visual_hv = hypervec_rs.HyperVector(seed)
        
        # Store the association
        self.concept_to_visual[concept] = visual_hv
        self.feature_templates[concept] = visual_features
        
    def retrieve_visual(self, concept: str) -> Optional[VisualFeatures]:
        """
        Retrieve visual features for a concept.
        
        Args:
            concept: The concept to query
            
        Returns:
            Visual features if found, None otherwise
        """
        return self.feature_templates.get(concept)
    
    def retrieve_similar_visual(self, concept_hv: Any, top_k: int = 3) -> List[Tuple[str, float, VisualFeatures]]:
        """
        Find most similar concepts by HV similarity.
        
        Args:
            concept_hv: Query hypervector
            top_k: Number of results to return
            
        Returns:
            List of (concept_name, similarity, visual_features)
        """
        similarities = []
        for concept, visual_hv in self.concept_to_visual.items():
            sim = concept_hv.similarity(visual_hv)
            if concept in self.feature_templates:
                similarities.append((concept, sim, self.feature_templates[concept]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class ImageGenerator:
    """
    VSA-based image generator that converts text to images without neural networks.
    """
    
    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        config: Optional[GenerationConfig] = None
    ):
        self.semantic = semantic_memory or SemanticMemory()
        self.config = config or GenerationConfig()
        
        # Initialize language cortex for text encoding
        self.lingua = get_lingua_cortex()
        
        # Concept-to-visual memory
        self.concept_memory = ConceptVisualMemory()
        
        # Statistics
        self.generation_count = 0
        
    def train_from_examples(
        self,
        text_image_pairs: List[Tuple[str, np.ndarray]],
        max_examples: Optional[int] = None
    ):
        """
        Train the generator from text-image pairs.
        
        Args:
            text_image_pairs: List of (text_description, image_array) tuples
            max_examples: Maximum number of examples to train on
        """
        from python.core.multimodal.multimodal_processor import MultimodalProcessor
        
        processor = MultimodalProcessor(None, self.semantic)
        
        print(f"Training image generator on {len(text_image_pairs)} examples...")
        
        for idx, (text, image) in enumerate(text_image_pairs[:max_examples] if max_examples else text_image_pairs):
            if idx % 100 == 0:
                print(f"  Processing example {idx}/{len(text_image_pairs)}")
            
            # Extract visual features from image
            visual_features = self._extract_visual_features(image, processor)
            
            # Extract concepts from text
            concepts = self._extract_concepts(text)
            
            # Learn associations
            for concept in concepts:
                # Get concept hypervector
                fp = self.lingua.get_fingerprint(concept)
                if fp is None:
                    # Learn the concept first
                    self.lingua.learn_text_snippet(concept)
                    fp = self.lingua.get_fingerprint(concept)
                
                if fp is not None:
                    concept_hv = self._fingerprint_to_hv(fp)
                    
                    # Store association
                    self.concept_memory.learn_association(concept, visual_features, concept_hv)
        
        print(f"Training complete! Learned {len(self.concept_memory.feature_templates)} concept-visual associations.")
    
    def _extract_visual_features(self, image: np.ndarray, processor: Any) -> VisualFeatures:
        """Extract visual features from an image using the multimodal processor."""
        from python.core.multimodal.multimodal_processor import MultimodalInput
        
        inp = MultimodalInput(image=image)
        result = processor.process(inp)
        
        # Extract features from the image modality result
        for mod_result in result.modality_results:
            if mod_result.modality == "image":
                stats = mod_result.features.get("stats", {})
                descriptors = mod_result.features.get("descriptors", [])
                
                # Extract numerical features from stats and descriptors
                # We'll use stats to reconstruct visual features
                edge_density = stats.get("edge_density", 0.5)
                mean_val = stats.get("mean", 128.0)
                std_val = stats.get("std", 40.0)
                is_color = stats.get("is_color", True)
                
                # Create approximate feature vectors
                # These are simplified since the processor doesn't return raw features
                rng = np.random.RandomState(int(mean_val + std_val * 100))
                
                return VisualFeatures(
                    hog_features=rng.rand(128) * edge_density,
                    color_features=self._make_color_features(mean_val, std_val, is_color),
                    edge_density=edge_density,
                    lbp_features=rng.rand(40) * std_val / 128.0,
                    spatial_features=self._make_spatial_features(mean_val, std_val)
                )
        
        # Fallback to random features
        return VisualFeatures.random()
    
    def _make_color_features(self, mean_val: float, std_val: float, is_color: bool) -> np.ndarray:
        """Create color histogram features from image statistics."""
        features = np.zeros(24)
        rng = np.random.RandomState(int(mean_val * 10))
        
        if is_color:
            # 3 channels × 8 bins
            for ch in range(3):
                # Create histogram centered around mean
                features[ch*8:(ch+1)*8] = rng.rand(8) * 0.3
                center_bin = int((mean_val / 255.0) * 7)
                features[ch*8 + center_bin] = 0.5
        else:
            # Grayscale
            features[:8] = rng.rand(8) * 0.3
            center_bin = int((mean_val / 255.0) * 7)
            features[center_bin] = 0.5
        
        # Normalize
        features = features / max(features.sum(), 1e-6)
        return features
    
    def _make_spatial_features(self, mean_val: float, std_val: float) -> np.ndarray:
        """Create spatial quadrant features."""
        features = np.zeros(8)
        rng = np.random.RandomState(int(std_val * 10))
        
        for i in range(4):
            features[i*2] = (mean_val / 255.0) + rng.randn() * 0.1
            features[i*2 + 1] = (std_val / 128.0) + rng.randn() * 0.05
        
        return np.clip(features, 0, 1)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple tokenization and filtering
        words = text.lower().split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}
        
        concepts = [w.strip('.,!?;:') for w in words if w not in stop_words and len(w) > 2]
        return concepts[:5]  # Limit to top 5 concepts
    
    def _fingerprint_to_hv(self, fp: SemanticFingerprint) -> Any:
        """Convert semantic fingerprint to hypervector."""
        # Use fingerprint bits as seed
        bits_flat = fp.bits.flatten()
        seed_str = ''.join(str(int(b)) for b in bits_flat[:100])  # Use first 100 bits
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        return hypervec_rs.HyperVector(seed)
    
    def generate(self, text_prompt: str) -> np.ndarray:
        """
        Generate an image from a text prompt.
        
        Args:
            text_prompt: Text description of the desired image
            
        Returns:
            Generated image as numpy array (H, W, C) for color or (H, W) for grayscale
        """
        print(f"Generating image for: '{text_prompt}'")
        
        # Extract concepts from prompt
        concepts = self._extract_concepts(text_prompt)
        print(f"  Extracted concepts: {concepts}")
        
        # Retrieve visual features for each concept
        all_features = []
        for concept in concepts:
            visual = self.concept_memory.retrieve_visual(concept)
            if visual:
                all_features.append(visual)
                print(f"    Found visual for: {concept}")
            else:
                # Try semantic similarity
                fp = self.lingua.get_fingerprint(concept)
                if fp is None:
                    # Try to learn the concept
                    self.lingua.learn_text_snippet(concept)
                    fp = self.lingua.get_fingerprint(concept)
                
                if fp is not None:
                    concept_hv = self._fingerprint_to_hv(fp)
                    similar = self.concept_memory.retrieve_similar_visual(concept_hv, top_k=1)
                    if similar:
                        sim_concept, sim_score, sim_visual = similar[0]
                        if sim_score > 0.3:  # Threshold for similarity
                            all_features.append(sim_visual)
                            print(f"    Using similar concept '{sim_concept}' (sim={sim_score:.2f})")
        
        if not all_features:
            print("  No visual features found, generating random image")
            image = self._generate_random_image()
            self.generation_count += 1
            return image
        
        # Bundle visual features (average)
        combined_features = self._bundle_features(all_features)
        
        # Decode features to image
        image = self._decode_features_to_image(combined_features)
        
        self.generation_count += 1
        return image
    
    def _bundle_features(self, features_list: List[VisualFeatures]) -> VisualFeatures:
        """Bundle multiple visual features into one by averaging."""
        if len(features_list) == 1:
            return features_list[0]
        
        # Average all features
        arrays = [f.to_array() for f in features_list]
        avg_array = np.mean(arrays, axis=0)
        
        return VisualFeatures.from_array(avg_array)
    
    def _decode_features_to_image(self, features: VisualFeatures) -> np.ndarray:
        """
        Decode visual features back to pixel space.
        This is the inverse of the feature extraction process.
        """
        h, w = self.config.image_size
        is_color = self.config.is_color
        
        # Initialize image
        if is_color:
            image = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            image = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Decode color features → pixel colors
        color_feat = features.color_features
        if is_color:
            # 8 bins per channel (RGB)
            for ch in range(3):
                channel_hist = color_feat[ch*8:(ch+1)*8]
                # Generate pixel values based on histogram
                pixels = self._histogram_to_pixels(channel_hist, h * w)
                image[:, :, ch] = pixels.reshape(h, w)
        else:
            # Grayscale
            gray_hist = color_feat[:8]
            pixels = self._histogram_to_pixels(gray_hist, h * w)
            image[:, :] = pixels.reshape(h, w)
        
        # 2. Apply spatial features → spatial variations
        spatial_feat = features.spatial_features
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            (0, mid_h, 0, mid_w),
            (0, mid_h, mid_w, w),
            (mid_h, h, 0, mid_w),
            (mid_h, h, mid_w, w)
        ]
        
        for qi, (rs, re, cs, ce) in enumerate(quadrants):
            mean_val = spatial_feat[qi * 2] * 255
            std_val = spatial_feat[qi * 2 + 1] * 128
            
            # Adjust quadrant
            if is_color:
                for ch in range(3):
                    quad = image[rs:re, cs:ce, ch].astype(np.float32)
                    quad = (quad - quad.mean()) * (std_val / max(quad.std(), 1.0)) + mean_val
                    image[rs:re, cs:ce, ch] = np.clip(quad, 0, 255).astype(np.uint8)
            else:
                quad = image[rs:re, cs:ce].astype(np.float32)
                quad = (quad - quad.mean()) * (std_val / max(quad.std(), 1.0)) + mean_val
                image[rs:re, cs:ce] = np.clip(quad, 0, 255).astype(np.uint8)
        
        # 3. Apply edge density → add edges
        if features.edge_density > 0.3:
            image = self._add_edges(image, features.edge_density)
        
        # 4. Apply smoothing
        if self.config.smoothing_factor > 0:
            image = self._smooth_image(image, self.config.smoothing_factor)
        
        # 5. Contrast boost
        if self.config.contrast_boost != 1.0:
            image = self._adjust_contrast(image, self.config.contrast_boost)
        
        return image
    
    def _histogram_to_pixels(self, histogram: np.ndarray, num_pixels: int) -> np.ndarray:
        """Generate pixel values from a histogram."""
        # Normalize histogram
        hist = histogram / max(histogram.sum(), 1e-6)
        
        # Generate pixel values based on histogram distribution
        bin_edges = np.linspace(0, 255, len(hist) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Sample from distribution
        rng = np.random.RandomState(int(hist.sum() * 1000) % 2**32)
        pixels = rng.choice(bin_centers, size=num_pixels, p=hist)
        
        return pixels.astype(np.uint8)
    
    def _add_edges(self, image: np.ndarray, edge_density: float) -> np.ndarray:
        """Add edge-like features to the image."""
        # Simple edge addition by local gradients
        if image.ndim == 3:
            gray = image.mean(axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Compute gradients (simple difference)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, :-1] = np.diff(gray, axis=1)
        gy[:-1, :] = np.diff(gray, axis=0)
        
        # Enhance edges
        magnitude = np.sqrt(gx**2 + gy**2)
        enhanced = gray + magnitude * edge_density * 0.5
        
        if image.ndim == 3:
            for ch in range(3):
                image[:, :, ch] = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            image = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return image
    
    def _smooth_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Apply simple smoothing (box filter)."""
        if factor <= 0:
            return image
        
        # Simple 3x3 box filter
        padded = np.pad(image, 1, mode='edge')
        if image.ndim == 3:
            smoothed = np.zeros_like(image, dtype=np.float32)
            for ch in range(3):
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        smoothed[i, j, ch] = padded[i:i+3, j:j+3, ch].mean()
        else:
            smoothed = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    smoothed[i, j] = padded[i:i+3, j:j+3].mean()
        
        # Blend original and smoothed
        result = (1 - factor) * image + factor * smoothed
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        if image.ndim == 3:
            mean = image.mean(axis=(0, 1), keepdims=True)
        else:
            mean = image.mean()
        
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def _generate_random_image(self) -> np.ndarray:
        """Generate a random image as fallback."""
        h, w = self.config.image_size
        rng = np.random.RandomState(self.generation_count)
        
        if self.config.is_color:
            return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            return rng.randint(0, 256, (h, w), dtype=np.uint8)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "generation_count": self.generation_count,
            "learned_concepts": len(self.concept_memory.feature_templates),
            "concept_list": list(self.concept_memory.feature_templates.keys())[:20],
            "config": {
                "image_size": self.config.image_size,
                "is_color": self.config.is_color,
                "smoothing_factor": self.config.smoothing_factor,
                "contrast_boost": self.config.contrast_boost
            }
        }


# Factory function
def get_image_generator(
    semantic_memory: Optional[SemanticMemory] = None,
    config: Optional[GenerationConfig] = None
) -> ImageGenerator:
    """Get or create an image generator instance."""
    return ImageGenerator(semantic_memory, config)


if __name__ == "__main__":
    # Quick test
    print("Testing NSCK Image Generator...")
    
    gen = ImageGenerator()
    
    # Generate a test image
    test_prompt = "a red cat with blue eyes"
    image = gen.generate(test_prompt)
    
    print(f"Generated image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min()}, {image.max()}]")
    print(f"\nStatistics: {gen.get_statistics()}")
