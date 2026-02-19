
import os
import torch
import logging
from typing import Iterator, Tuple, Optional

# Try imports, handle missing dependencies gracefully
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False

try:
    from torchvision import transforms
    from PIL import Image
    HAVE_VISION = True
except ImportError:
    HAVE_VISION = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HF_Adapter")

class HuggingFaceAdapter:
    """
    Bridge between Hugging Face Datasets and NSCK Cognitive System.
    Supports streaming to minimize RAM usage.
    """
    
    def __init__(self, cache_dir: str = "./hf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.transform = None
        
        if HAVE_VISION:
            # Standard ImageNet/CIFAR normalization
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)), # Encoder expects 32x32 usually or resize internally
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

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

    def stream_vision_data(self, dataset_name="cifar10", split="train", limit=1000) -> Iterator[Tuple[torch.Tensor, int]]:
        """
        Yields (image_tensor, label) from CIFAR-10.
        """
        if not HAVE_DATASETS or not HAVE_VISION:
            logger.warning("datasets or torchvision not found. Yielding mock noise.")
            for _ in range(10):
                yield torch.randn(3, 32, 32), 0
            return

        try:
            logger.info(f"Streaming {dataset_name} split={split}...")
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            count = 0
            for item in dataset:
                image = item['img'] # PIL Image
                label = item['label']
                
                if self.transform:
                    image_t = self.transform(image)
                else:
                    # Fallback to simple tensor conversion
                    import numpy as np
                    arr = np.array(image).transpose(2, 0, 1) / 255.0
                    image_t = torch.tensor(arr, dtype=torch.float32)

                yield image_t, label
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Failed to load vision dataset: {e}")
            # Mock
            yield torch.randn(3, 32, 32), 0

# Validation Block
if __name__ == "__main__":
    adapter = HuggingFaceAdapter()
    
    print("--- Text Stream Test ---")
    for i, text in enumerate(adapter.stream_text_data(limit=3)):
        print(f"Sample {i}: {text[:50]}...")
        
    print("\n--- Vision Stream Test ---")
    for i, (img, lbl) in enumerate(adapter.stream_vision_data(limit=3)):
        print(f"Sample {i}: Shape {img.shape}, Label {lbl}")
