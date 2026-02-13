
import os
import logging
from typing import Iterator, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HF_Adapter")

# Try imports, handle missing dependencies gracefully
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False
    logger.warning("datasets library not found - streaming will use mock data")

# Lazy imports for torch/torchvision - avoid heavy initialization at import time
HAVE_VISION = False
HAVE_TORCH = False
transforms = None
Image = None
torch = None

def ensure_vision():
    """Lazy-load vision dependencies."""
    global HAVE_VISION, transforms, Image, torch
    if HAVE_VISION:
        return True
    try:
        import torch as _torch
        from torchvision import transforms as _transforms
        from PIL import Image as _Image
        torch = _torch
        transforms = _transforms
        Image = _Image
        HAVE_VISION = True
        return True
    except ImportError as e:
        logger.warning(f"Vision imports failed: {e}")
        return False

def ensure_torch():
    """Lazy-load torch if needed."""
    global HAVE_TORCH, torch
    if HAVE_TORCH:
        return True
    try:
        import torch as _torch
        torch = _torch
        HAVE_TORCH = True
        return True
    except ImportError:
        return False


class HuggingFaceAdapter:
    """
    Bridge between Hugging Face Datasets and NSCK Cognitive System.
    Supports streaming to minimize RAM usage.
    """
    
    def __init__(self, cache_dir: str = "./hf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.transform = None
        
        # Lazy-load vision if available
        if ensure_vision():
            try:
                # Standard ImageNet/CIFAR normalization
                self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            except Exception as e:
                logger.warning(f"Failed to create transforms: {e}")

    def stream_text_data(self, dataset_name="wikitext", config="wikitext-2-v1", split="train", limit=1000) -> Iterator[str]:
        """
        Yields text samples from WikiText.
        """
        if not HAVE_DATASETS:
            logger.warning("datasets library not found. Yielding mock text.")
            yield "The quick brown fox jumps over the lazy dog."
            yield "NSCK is a neuro-symbolic cognitive architecture."
            return

        try:
            logger.info(f"Streaming {dataset_name}/{config} split={split}...")
            # Stream = True avoids downloading the whole 1GB+ dataset if it's large
            dataset = load_dataset(dataset_name, config, split=split, streaming=True)
            
            count = 0
            for item in dataset:
                text = item.get('text', '').strip()
                if len(text) > 20: # Skip headers/empty lines
                    yield text
                    count += 1
                    if count >= limit:
                        break
        except Exception as e:
            logger.error(f"Failed to load text dataset: {e}")
            yield "Error loading dataset. Please check internet connection."

    def stream_vision_data(self, dataset_name="cifar10", split="train", limit=1000) -> Iterator[Tuple]:
        """
        Yields (image_tensor, label) from CIFAR-10.
        """
        if not HAVE_DATASETS:
            logger.warning("datasets not found. Yielding mock noise.")
            ensure_torch()
            if torch:
                for i in range(min(10, limit)):
                    yield torch.randn(3, 32, 32), 0
            return
        
        if not ensure_vision():
            logger.warning("Vision libraries not available. Yielding mock noise.")
            import numpy as np
            for i in range(min(10, limit)):
                yield np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8), 0
            return

        try:
            logger.info(f"Streaming {dataset_name} split={split}...")
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            count = 0
            for item in dataset:
                try:
                    image = item['img'] if 'img' in item else item.get('image')
                    label = item['label']
                    
                    if image is None:
                        continue
                    
                    if self.transform:
                        image_t = self.transform(image)
                    else:
                        # Fallback to simple tensor conversion
                        import numpy as np
                        if hasattr(image, 'convert'):
                            image = image.convert('RGB')
                        arr = np.array(image).astype(np.float32)
                        if arr.dtype != np.uint8 and arr.max() > 1:
                            arr = arr / 255.0
                        if len(arr.shape) == 3:
                            arr = arr.transpose(2, 0, 1)
                        if torch:
                            image_t = torch.tensor(arr, dtype=torch.float32)
                        else:
                            image_t = arr
                    
                    yield image_t, label
                    count += 1
                    if count >= limit:
                        break
                except Exception as item_err:
                    logger.debug(f"Skipping item: {item_err}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load vision dataset: {e}")
            # Yield a few mock samples as fallback
            import numpy as np
            for _ in range(min(3, limit)):
                yield np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8), 0

# Validation Block
if __name__ == "__main__":
    adapter = HuggingFaceAdapter()
    
    print("--- Text Stream Test ---")
    for i, text in enumerate(adapter.stream_text_data(limit=3)):
        print(f"Sample {i}: {text[:50]}...")
        
    print("\n--- Vision Stream Test ---")
    for i, (img, lbl) in enumerate(adapter.stream_vision_data(limit=3)):
        print(f"Sample {i}: Shape {img.shape}, Label {lbl}")
