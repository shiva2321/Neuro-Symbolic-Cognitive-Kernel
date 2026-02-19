
import os
import sys
from typing import Iterator, Optional, List, Dict, Any
import datasets
import re

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TextStreamer:
    """
    Streams text data from Hugging Face datasets for VSA training.
    Optimized for memory efficiency (no full download).
    """
    
    # Common datasets known to work well
    DATASETS = {
        "wikitext": {"path": "wikitext", "name": "wikitext-2-v1", "split": "train", "text_col": "text"},
        "tinystories": {"path": "roneneldan/TinyStories", "name": None, "split": "train", "text_col": "text"},
        "c4": {"path": "c4", "name": "en", "split": "train", "text_col": "text", "streaming": True},
    }

    def __init__(self, dataset_name: str = "wikitext", split: str = "train"):
        """
        Initialize the streamer.
        Args:
            dataset_name: 'wikitext', 'tinystories', or other HF dataset path.
            split: 'train', 'validation', 'test'
        """
        self.config = self.DATASETS.get(dataset_name, {
            "path": dataset_name, 
            "name": None, 
            "split": split, 
            "text_col": "text"
        })
        
        # Allow override of split if provided explicitly
        if split != "train":
            self.config["split"] = split
            
        self.dataset = None
        
    def load(self):
        """Lazy load the dataset in streaming mode."""
        print(f"[Streamer] Loading {self.config['path']} ({self.config.get('name')}) split='{self.config['split']}' in streaming mode...")
        try:
            self.dataset = datasets.load_dataset(
                self.config["path"], 
                self.config["name"], 
                split=self.config["split"], 
                streaming=True
            )
        except Exception as e:
            print(f"[Streamer] Error loading dataset: {e}")
            raise

    def stream_sentences(self, limit: Optional[int] = None) -> Iterator[str]:
        """
        Yields clean, non-empty sentences from the dataset.
        """
        if not self.dataset:
            self.load()
            
        count = 0
        text_col = self.config.get("text_col", "text")
        
        # Simple sentence splitter
        # We start with a basic regex, but could be swapped for NLTK if needed
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        
        for sample in self.dataset:
            text = sample.get(text_col, "").strip()
            
            # Basic cleaning
            if not text or len(text) < 10:
                continue
                
            # Handle WikiText specific artifacts
            if text.startswith("=") or text.startswith(" @"):
                continue

            # Split into sentences
            # text.split('\n') is okay for some, but proper segmentation is better.
            # Let's try to handle both newlines and punctuation.
            segments = sentence_endings.split(text)
            
            for seg in segments:
                # Clean up segment
                sent = seg.strip().replace('\n', ' ')
                
                # Filter out garbage
                if len(sent) < 10: continue
                if sent.startswith("="): continue
                if len(sent.split()) < 3: continue # Too short to be a sentence
                
                yield sent
                count += 1
                if limit and count >= limit:
                    return

    def stream_batches(self, batch_size: int = 10, limit: Optional[int] = None) -> Iterator[List[str]]:
        """
        Yields batches of sentences.
        """
        batch = []
        count = 0
        
        for sentence in self.stream_sentences(limit=limit):
            batch.append(sentence)
            if len(batch) >= batch_size:
                yield batch
                batch = []
            
            count += 1
            if limit and count >= limit:
                break
                
        if batch:
            yield batch

if __name__ == "__main__":
    # Test
    import time
    
    print("Test 1: WikiText")
    streamer = TextStreamer("wikitext")
    count = 0
    start = time.time()
    for s in streamer.stream_sentences(limit=5):
        print(f" -> {s}")
        count += 1
    print(f"Done in {time.time()-start:.2f}s")
    
    print("\nTest 2: TinyStories (if available, else handled gracefully)")
    try:
        ts_streamer = TextStreamer("tinystories")
        for s in ts_streamer.stream_sentences(limit=3):
            print(f" -> [Tiny] {s}")
    except Exception as e:
        print(f"TinyStories skipped: {e}")
