"""
NSCK AI Data Pipeline — HuggingFace Data Loading & Preprocessing
================================================================

What this does:
    Streams real text data from HuggingFace datasets (WikiText, etc.)
    and feeds it into the NSCK AI engine for training.

How it works:
    1. Uses the ``datasets`` library to stream text from HuggingFace.
    2. Cleans and segments text into sentence-level passages.
    3. Yields passages to the engine's ``train_on_text()`` method.
    4. Falls back to a built-in corpus if no internet / no ``datasets``.

Why streaming?
    WikiText-2 alone is ~2 million tokens.  Streaming avoids downloading
    the entire dataset into RAM.  Each sample is processed and discarded.
"""

import os
import re
import logging
from typing import Iterator, Dict, Any, Optional

logger = logging.getLogger("nsck_ai.data")

# Try to import HuggingFace datasets
try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ── Built-in seed corpus ─────────────────────────────────────────────────
# This is NOT hardcoded knowledge — it is *training data* that the engine
# processes through its normal learning pipeline, the same way it would
# process WikiText or any other corpus.  These sentences bootstrap the
# n-gram model so the engine can generate grammatical text even before
# HuggingFace data arrives.

_SEED_CORPUS = [
    # Geography
    "Paris is the capital of France.",
    "France is a country in Europe.",
    "Berlin is the capital of Germany.",
    "Germany is a country in Europe.",
    "Tokyo is the capital of Japan.",
    "Japan is a country in Asia.",
    "London is the capital of the United Kingdom.",
    "The United Kingdom is a country in Europe.",
    "Washington is the capital of the United States.",
    "The United States is a country in North America.",
    "Cairo is the capital of Egypt.",
    "Egypt is a country in Africa.",
    "Beijing is the capital of China.",
    "China is a country in Asia.",
    "Moscow is the capital of Russia.",

    # Science
    "Water is a molecule made of hydrogen and oxygen.",
    "The Earth orbits the Sun.",
    "The Moon orbits the Earth.",
    "The Sun is a star.",
    "Plants use photosynthesis to convert sunlight into energy.",
    "Gravity is a force that attracts objects toward each other.",
    "Light travels at approximately 300,000 kilometres per second.",
    "DNA carries genetic information in living organisms.",
    "Atoms are the basic building blocks of matter.",
    "The speed of sound in air is about 343 metres per second.",
    "Electrons orbit the nucleus of an atom.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The human body has 206 bones.",
    "Oxygen is essential for respiration in most living organisms.",

    # Animals
    "Dogs are mammals.",
    "Cats are mammals.",
    "Whales are mammals.",
    "Eagles are birds.",
    "Salmon are fish.",
    "Frogs are amphibians.",
    "Snakes are reptiles.",
    "Spiders are arachnids.",
    "Bees produce honey.",
    "Dolphins are highly intelligent marine mammals.",

    # Technology
    "Python is a programming language.",
    "Java is a programming language.",
    "JavaScript is a programming language.",
    "The internet connects computers around the world.",
    "A computer processor executes instructions.",
    "Machine learning is a subfield of artificial intelligence.",
    "Databases store and organize data.",
    "Encryption protects data from unauthorized access.",
    "An operating system manages computer hardware and software.",
    "Cloud computing provides computing resources over the internet.",

    # Cause and effect
    "Rain causes flooding in low-lying areas.",
    "Deforestation causes soil erosion.",
    "Exercise improves cardiovascular health.",
    "Pollution causes environmental damage.",
    "Education leads to better employment opportunities.",
    "Smoking causes lung cancer.",
    "Vaccination prevents infectious diseases.",
    "Overheating causes metal to expand.",
    "Lack of sleep causes fatigue and poor concentration.",

    # Culture and history
    "Shakespeare wrote plays and sonnets.",
    "The Renaissance began in Italy in the 14th century.",
    "Democracy is a system of government.",
    "Music is a form of artistic expression.",
    "Mathematics is the study of numbers and patterns.",
    "Philosophy explores questions about existence and knowledge.",
    "The printing press was invented by Johannes Gutenberg.",
    "The Industrial Revolution began in Britain in the 18th century.",
]


def stream_seed_corpus() -> Iterator[str]:
    """Yield sentences from the built-in seed corpus."""
    for s in _SEED_CORPUS:
        yield s


def stream_hf_text(dataset_name: str = "wikitext",
                   config: str = "wikitext-2-v1",
                   split: str = "train",
                   limit: int = 2000) -> Iterator[str]:
    """Stream text from a HuggingFace dataset.

    Yields individual paragraphs (blocks of ≥ 30 characters).
    Falls back to the seed corpus if ``datasets`` is not installed
    or the download fails.
    """
    if not _HF_AVAILABLE:
        logger.warning("datasets library not available; using seed corpus")
        yield from stream_seed_corpus()
        return

    try:
        logger.info("Streaming %s/%s split=%s (limit=%d)…",
                     dataset_name, config, split, limit)
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
        count = 0
        for item in ds:
            text = item.get('text', '').strip()
            # Skip WikiText section headers and very short lines
            if len(text) < 30:
                continue
            if text.startswith('='):
                continue
            yield text
            count += 1
            if count >= limit:
                break
        logger.info("Streamed %d passages from %s", count, dataset_name)
    except Exception as exc:
        logger.error("Failed to stream %s: %s — falling back to seed corpus",
                     dataset_name, exc)
        yield from stream_seed_corpus()


def stream_general_knowledge(limit: int = 2000) -> Iterator[str]:
    """Stream a mix of seed corpus + WikiText for general knowledge."""
    # Always start with the seed corpus for baseline knowledge
    yield from stream_seed_corpus()
    # Then stream from WikiText
    yield from stream_hf_text(limit=limit)


class DataPipeline:
    """Manages data loading and feeding into the engine.

    Usage::

        from nsck_ai_model.ai_engine import NSCKAIEngine
        from nsck_ai_model.data_pipeline import DataPipeline

        engine = NSCKAIEngine()
        pipeline = DataPipeline(engine)
        stats = pipeline.train_from_seed()            # fast, local
        stats = pipeline.train_from_hf(limit=500)     # real HF data
    """

    def __init__(self, engine):
        self.engine = engine
        self._total_passages = 0
        self._total_time = 0.0

    def train_from_seed(self) -> Dict[str, Any]:
        """Train the engine on the built-in seed corpus."""
        import time
        t0 = time.time()
        count = 0
        for text in stream_seed_corpus():
            self.engine.train_on_text(text)
            count += 1
        elapsed = time.time() - t0
        self._total_passages += count
        self._total_time += elapsed
        logger.info("Seed training: %d passages in %.2fs", count, elapsed)
        return {'passages': count, 'elapsed_s': round(elapsed, 2),
                'source': 'seed_corpus'}

    def train_from_hf(self, dataset_name: str = "wikitext",
                      config: str = "wikitext-2-v1",
                      split: str = "train",
                      limit: int = 500) -> Dict[str, Any]:
        """Train the engine on HuggingFace data."""
        import time
        t0 = time.time()
        count = 0
        for text in stream_hf_text(dataset_name, config, split, limit):
            self.engine.train_on_text(text)
            count += 1
        elapsed = time.time() - t0
        self._total_passages += count
        self._total_time += elapsed
        logger.info("HF training: %d passages in %.2fs", count, elapsed)
        return {'passages': count, 'elapsed_s': round(elapsed, 2),
                'source': f'{dataset_name}/{config}'}

    def train_general(self, hf_limit: int = 500) -> Dict[str, Any]:
        """Train on seed corpus + HuggingFace data."""
        import time
        t0 = time.time()
        count = 0
        for text in stream_general_knowledge(limit=hf_limit):
            self.engine.train_on_text(text)
            count += 1
        elapsed = time.time() - t0
        self._total_passages += count
        self._total_time += elapsed
        return {'passages': count, 'elapsed_s': round(elapsed, 2),
                'source': 'seed+wikitext'}

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_passages': self._total_passages,
            'total_time_s': round(self._total_time, 2),
        }
