"""
NSCK Image Understanding - VSA-based image encoding
===================================================

Encodes images into hypervectors for multimodal understanding.
Uses classical CV features (no neural networks).
"""

import sys
import os
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List

# Add nsck-demo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
    HyperVector = hypervec_rs.HyperVector
except ImportError:
    from python.core.vsa.hypervec_py import HyperVector

# Try to import PIL
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class ImageEncoder:
    """Encode images to hypervectors using classical CV features."""
    
    def __init__(self, grid_size=4, color_bins=16):
        self.grid_size = grid_size
        self.color_bins = color_bins
    
    @staticmethod
    def _bundle_hvs(hvs):
        """Bundle a list of hypervectors."""
        if not hvs:
            return HyperVector(42)
        if len(hvs) == 1:
            return hvs[0]
        
        # Recursively bundle pairs
        result = hvs[0]
        for hv in hvs[1:]:
            result = result.bundle(hv)
        return result
    
    def encode_image(self, image):
        """Encode an image to a hypervector."""
        # Convert PIL Image to numpy if needed
        if PIL_AVAILABLE and isinstance(image, PILImage.Image):
            arr = np.array(image)
        else:
            arr = image
        
        # Ensure 3D array (H, W, C)
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        
        # Extract features and encode to HVs
        color_hv = self._color_histogram_hv(arr)
        edge_hv = self._edge_histogram_hv(arr)
        spatial_hv = self._spatial_layout_hv(arr)
        
        # Bundle all features
        image_hv = self._bundle_hvs([color_hv, edge_hv, spatial_hv])
        return image_hv
    
    def _color_histogram_hv(self, arr):
        """Encode color histogram."""
        hvs = []
        for channel in range(min(3, arr.shape[2])):
            hist, _ = np.histogram(arr[:, :, channel], bins=self.color_bins, range=(0, 256))
            for bin_idx, count in enumerate(hist):
                if count > 0:
                    # Create HV for this bin
                    seed = hash(f"color_ch{channel}_bin{bin_idx}") % (2**31)
                    bin_hv = HyperVector(seed)
                    # Weight by count (repeated bundling)
                    weight = min(int(count / 10), 5)  # Limit weight
                    hvs.extend([bin_hv] * weight)
        
        return self._bundle_hvs(hvs)
    
    def _edge_histogram_hv(self, arr):
        """Encode edge information using simple gradients."""
        # Convert to grayscale
        if len(arr.shape) == 3:
            gray = arr.mean(axis=2).astype(np.float32)
        else:
            gray = arr.astype(np.float32)
        
        # Simple gradient (Sobel-like)
        dy = np.abs(np.diff(gray, axis=0))
        dx = np.abs(np.diff(gray, axis=1))
        
        # Edge magnitude histogram
        edges = np.concatenate([dy.flatten(), dx.flatten()])
        hist, _ = np.histogram(edges, bins=8, range=(0, 256))
        
        hvs = []
        for bin_idx, count in enumerate(hist):
            if count > 0:
                seed = hash(f"edge_bin{bin_idx}") % (2**31)
                bin_hv = HyperVector(seed)
                weight = min(int(count / 100), 5)
                hvs.extend([bin_hv] * weight)
        
        return self._bundle_hvs(hvs)
    
    def _spatial_layout_hv(self, arr):
        """Encode spatial layout using grid."""
        h, w = arr.shape[:2]
        cell_h = max(1, h // self.grid_size)
        cell_w = max(1, w // self.grid_size)
        
        hvs = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y0 = row * cell_h
                x0 = col * cell_w
                y1 = min(y0 + cell_h, h)
                x1 = min(x0 + cell_w, w)
                
                cell = arr[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                
                # Mean color in this cell
                mean_color = cell.mean(axis=(0, 1))
                
                # Quantize color
                qr = int(mean_color[0] / 32) if len(mean_color) > 0 else 0
                qg = int(mean_color[1] / 32) if len(mean_color) > 1 else 0
                qb = int(mean_color[2] / 32) if len(mean_color) > 2 else 0
                
                # Create HV for this spatial cell
                seed = hash(f"spatial_r{row}_c{col}_rgb{qr}{qg}{qb}") % (2**31)
                cell_hv = HyperVector(seed)
                hvs.append(cell_hv)
        
        return self._bundle_hvs(hvs)


class ImageUnderstanding:
    """Cross-modal image-text understanding using VSA."""
    
    def __init__(self, backend_wrapper):
        """
        backend_wrapper should have:
        - encoder: with encode_sentence() and extract_concepts()
        - backend: with semantic memory
        - train_on_text(text): method to learn text
        """
        self.backend = backend_wrapper
        self.image_encoder = ImageEncoder()
        self.image_store = {}  # Store learned images
    
    def learn_image(self, image, description, image_id=None):
        """Learn an image with its text description."""
        if image_id is None:
            image_id = hashlib.md5(description.encode()).hexdigest()[:8]
        
        # Encode image
        img_hv = self.image_encoder.encode_image(image)
        
        # Encode text description
        # Use text learner's concept extractor
        concepts = []
        words = description.lower().split()
        for word in words:
            if len(word) > 2:  # Skip short words
                concepts.append(word)
        
        # Encode description using direct word hashing
        text_hvs = []
        for word in concepts[:5]:
            word_hv = HyperVector(hash(word) % (2**31))
            text_hvs.append(word_hv)
        text_hv = ImageEncoder._bundle_hvs(text_hvs)
        
        # Create joint representation
        joint_hv = img_hv.bundle(text_hv)
        
        # Store
        self.image_store[image_id] = {
            'description': description,
            'image_hv': img_hv,
            'text_hv': text_hv,
            'joint_hv': joint_hv,
            'concepts': concepts,
        }
        
        # Also train backend on description
        try:
            self.backend.train_on_text(description)
        except:
            pass
        
        return {
            'image_id': image_id,
            'concepts': concepts,
            'description': description,
        }
    
    def search_by_text(self, query, top_k=5):
        """Find images matching a text query."""
        # Encode query using direct word hashing
        words = query.lower().split()
        query_hvs = []
        for word in words:
            if len(word) > 2:
                word_hv = HyperVector(hash(word) % (2**31))
                query_hvs.append(word_hv)
        query_hv = ImageEncoder._bundle_hvs(query_hvs)
        
        # Search image store
        results = []
        for img_id, data in self.image_store.items():
            # Compare with text HV (text-to-text match)
            text_sim = query_hv.similarity(data['text_hv'])
            # Also check joint representation
            joint_sim = query_hv.similarity(data['joint_hv'])
            
            sim = max(text_sim, joint_sim)
            
            if sim > 0.3:  # Threshold
                results.append({
                    'image_id': img_id,
                    'description': data['description'],
                    'similarity': round(sim, 3),
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def search_by_image(self, image, top_k=5):
        """Find stored images similar to a query image."""
        query_hv = self.image_encoder.encode_image(image)
        
        results = []
        for img_id, data in self.image_store.items():
            sim = query_hv.similarity(data['image_hv'])
            if sim > 0.3:
                results.append({
                    'image_id': img_id,
                    'description': data['description'],
                    'similarity': round(sim, 3),
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def describe_image(self, image):
        """Describe an image by finding similar stored images."""
        results = self.search_by_image(image, top_k=1)
        if results:
            return results[0]['description']
        return "I have not learned enough about images yet to describe this."
    
    def get_stats(self):
        """Get statistics about stored images."""
        return {
            'images_stored': len(self.image_store),
            'image_ids': list(self.image_store.keys())[:10],
        }
