"""
NSCK Autonomous Data Fetcher
=============================

Fetches real-world datasets from HuggingFace and the web for training
the NSCK AI model on text, image, math, reasoning, and conversation data.

Datasets sourced:
    Text:        WikiText-2, TinyStories, BookCorpus subsets
    Knowledge:   OpenBookQA, ARC (AI2 Reasoning Challenge)
    Math:        GSM8K (grade school math), MATH subsets
    Conversation: Dolly-15k, OpenAssistant
    Image-text:  CIFAR-10 labels, Flickr8k captions
    Reasoning:   HellaSwag, PIQA, WinoGrande

All datasets are streamed to avoid large disk usage.
No neural network preprocessing — raw text/images only.
"""

import os
import re
import sys
import time
import json
import random
import hashlib
import logging
import signal
import threading
from typing import Iterator, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np

logger = logging.getLogger("nsck_ai.data_fetcher")

# ---------------------------------------------------------------------------
# Try to import optional dependencies
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning("datasets library not available")


def _load_dataset_with_timeout(
    *args, timeout_s: int = 30, **kwargs
):
    """Call ``load_dataset`` with a timeout.

    HuggingFace streaming dataset initialisation involves HTTP calls that
    can hang indefinitely when the Hub is slow or unreachable.  This
    wrapper uses a background thread so we can abort if it takes too long.
    """
    result = [None]
    exc = [None]

    def _load():
        try:
            result[0] = load_dataset(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise TimeoutError(
            f"load_dataset({args}) timed out after {timeout_s}s")
    if exc[0] is not None:
        raise exc[0]
    return result[0]

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TextSample:
    """A single text training sample."""
    text: str
    source: str
    category: str = "general"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ImageTextSample:
    """An image-text pair for multimodal training."""
    image: np.ndarray          # H×W×C uint8
    caption: str
    source: str
    label: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QASample:
    """A question-answer pair for evaluation and training."""
    question: str
    answer: str
    choices: List[str] = None
    source: str = ""
    category: str = "general"
    difficulty: str = "medium"

    def __post_init__(self):
        if self.choices is None:
            self.choices = []


@dataclass
class MathSample:
    """A math problem with solution."""
    question: str
    answer: str
    solution_steps: str = ""
    source: str = ""
    difficulty: str = "easy"


@dataclass
class ConversationSample:
    """A multi-turn conversation sample."""
    turns: List[Dict[str, str]]     # [{"role": "user", "content": ...}, ...]
    source: str = ""
    category: str = "general"


# ═══════════════════════════════════════════════════════════════════════════
# Text Data Sources
# ═══════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Clean a text passage for training."""
    text = text.strip()
    # Remove Wikipedia-style headers
    text = re.sub(r'^=+.*=+$', '', text, flags=re.MULTILINE)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove very short lines
    lines = [l for l in text.split('\n') if len(l.strip()) > 10]
    text = '\n'.join(lines)
    return text.strip()


def _split_into_passages(text: str, max_len: int = 500) -> List[str]:
    """Split a long text into training-sized passages."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    passages = []
    current = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > max_len and current:
            passages.append(' '.join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        passages.append(' '.join(current))

    return [p for p in passages if len(p) > 30]


def _iter_with_stall_guard(iterator, per_item_timeout: float = 10.0):
    """Wrap an iterator so that if any single ``next()`` call takes
    longer than *per_item_timeout* seconds we abort gracefully instead
    of hanging the whole training run.
    """
    it = iter(iterator)
    while True:
        result = [None]
        done = [False]
        exc = [None]

        def _fetch():
            try:
                result[0] = next(it)
            except StopIteration:
                done[0] = True
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=per_item_timeout)

        if t.is_alive():
            logger.warning("Dataset iterator stalled (>%.0fs) — aborting",
                           per_item_timeout)
            return
        if done[0]:
            return
        if exc[0] is not None:
            raise exc[0]
        yield result[0]


def fetch_wikitext(limit: int = 5000) -> Iterator[TextSample]:
    """Fetch WikiText-2 passages — encyclopedic knowledge."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("wikitext", "wikitext-2-v1",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            text = _clean_text(item.get('text', ''))
            if len(text) < 40:
                continue
            for passage in _split_into_passages(text):
                yield TextSample(
                    text=passage, source="wikitext-2",
                    category="encyclopedia")
                count += 1
                if count >= limit:
                    return
    except Exception as e:
        logger.warning("WikiText fetch failed: %s", e)


def fetch_tinystories(limit: int = 3000) -> Iterator[TextSample]:
    """Fetch TinyStories — simple narratives for language learning."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("roneneldan/TinyStories",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            text = item.get('text', '').strip()
            if len(text) < 50:
                continue
            # TinyStories are already passage-sized
            yield TextSample(
                text=text[:800], source="tinystories",
                category="narrative")
            count += 1
            if count >= limit:
                return
    except Exception as e:
        logger.warning("TinyStories fetch failed: %s", e)


def fetch_dolly_conversations(limit: int = 2000) -> Iterator[TextSample]:
    """Fetch Dolly-15k — instruction-following conversations."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("databricks/databricks-dolly-15k",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            instruction = item.get('instruction', '').strip()
            context = item.get('context', '').strip()
            response = item.get('response', '').strip()
            category = item.get('category', 'general')

            if not instruction or not response:
                continue

            # Learn the response as text
            full_text = response
            if context:
                full_text = f"{context} {response}"

            yield TextSample(
                text=full_text, source="dolly-15k",
                category=category,
                metadata={"instruction": instruction})
            count += 1
            if count >= limit:
                return
    except Exception as e:
        logger.warning("Dolly fetch failed: %s", e)


def fetch_conversations(limit: int = 1000) -> Iterator[ConversationSample]:
    """Fetch multi-turn conversations from Dolly-15k."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("databricks/databricks-dolly-15k",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            instruction = item.get('instruction', '').strip()
            response = item.get('response', '').strip()
            category = item.get('category', 'general')
            if not instruction or not response:
                continue
            turns = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            yield ConversationSample(
                turns=turns, source="dolly-15k", category=category)
            count += 1
            if count >= limit:
                return
    except Exception as e:
        logger.warning("Conversations fetch failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge & Reasoning Data Sources
# ═══════════════════════════════════════════════════════════════════════════

def fetch_openbookqa(limit: int = 500) -> Iterator[QASample]:
    """Fetch OpenBookQA — science knowledge questions."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("allenai/openbookqa", "main",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            question = item.get('question_stem', '').strip()
            choices_data = item.get('choices', {})
            answer_key = item.get('answerKey', '')

            labels = choices_data.get('label', [])
            texts = choices_data.get('text', [])
            choices = [f"{l}. {t}" for l, t in zip(labels, texts)]

            answer_idx = labels.index(answer_key) if answer_key in labels else 0
            answer = texts[answer_idx] if answer_idx < len(texts) else ""

            if question and answer:
                yield QASample(
                    question=question, answer=answer,
                    choices=choices, source="openbookqa",
                    category="science")
                count += 1
                if count >= limit:
                    return
    except Exception as e:
        logger.warning("OpenBookQA fetch failed: %s", e)


def fetch_arc(limit: int = 500) -> Iterator[QASample]:
    """Fetch ARC (AI2 Reasoning Challenge) — science reasoning."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("allenai/ai2_arc", "ARC-Easy",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            question = item.get('question', '').strip()
            choices_data = item.get('choices', {})
            answer_key = item.get('answerKey', '')

            labels = choices_data.get('label', [])
            texts = choices_data.get('text', [])
            choices = [f"{l}. {t}" for l, t in zip(labels, texts)]

            answer_idx = labels.index(answer_key) if answer_key in labels else 0
            answer = texts[answer_idx] if answer_idx < len(texts) else ""

            if question and answer:
                yield QASample(
                    question=question, answer=answer,
                    choices=choices, source="arc-easy",
                    category="science", difficulty="medium")
                count += 1
                if count >= limit:
                    return
    except Exception as e:
        logger.warning("ARC fetch failed: %s", e)


def fetch_piqa(limit: int = 500) -> Iterator[QASample]:
    """Fetch PIQA — physical intuition questions."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("ybisk/piqa", split="train",
                          streaming=True, trust_remote_code=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            goal = item.get('goal', '').strip()
            sol1 = item.get('sol1', '').strip()
            sol2 = item.get('sol2', '').strip()
            label = item.get('label', 0)

            answer = sol1 if label == 0 else sol2
            if goal and answer:
                yield QASample(
                    question=goal, answer=answer,
                    choices=[sol1, sol2], source="piqa",
                    category="physical_intuition")
                count += 1
                if count >= limit:
                    return
    except Exception as e:
        logger.warning("PIQA fetch failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# Math Data Sources
# ═══════════════════════════════════════════════════════════════════════════

def fetch_gsm8k(limit: int = 500) -> Iterator[MathSample]:
    """Fetch GSM8K — grade school math word problems."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("openai/gsm8k", "main",
                          split="train", streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            question = item.get('question', '').strip()
            answer_text = item.get('answer', '').strip()

            # Extract the final numeric answer after ####
            final_answer = ""
            if '####' in answer_text:
                parts = answer_text.split('####')
                solution_steps = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                solution_steps = answer_text
                # Try to extract last number
                nums = re.findall(r'-?\d+\.?\d*', answer_text)
                if nums:
                    final_answer = nums[-1]

            if question and final_answer:
                yield MathSample(
                    question=question, answer=final_answer,
                    solution_steps=solution_steps, source="gsm8k",
                    difficulty="grade_school")
                count += 1
                if count >= limit:
                    return
    except Exception as e:
        logger.warning("GSM8K fetch failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# Image-Text Data Sources
# ═══════════════════════════════════════════════════════════════════════════

# CIFAR-10 label descriptions for training image-text associations
_CIFAR10_DESCRIPTIONS = {
    0: "An airplane flying in the sky.",
    1: "An automobile or car on the road.",
    2: "A bird with feathers and wings.",
    3: "A cat with whiskers and fur.",
    4: "A deer in a natural habitat.",
    5: "A dog with fur and a tail.",
    6: "A frog near water.",
    7: "A horse running in a field.",
    8: "A ship sailing on the ocean.",
    9: "A truck on the highway.",
}


def fetch_cifar10_pairs(limit: int = 1000) -> Iterator[ImageTextSample]:
    """Fetch CIFAR-10 images with text descriptions."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("uoft-cs/cifar10", split="train",
                          streaming=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            label = item.get('label', -1)
            img = item.get('img', None)

            if img is None or label not in _CIFAR10_DESCRIPTIONS:
                continue

            # Convert PIL Image to numpy array
            if hasattr(img, 'convert'):
                img_arr = np.array(img.convert('RGB'))
            else:
                continue

            caption = _CIFAR10_DESCRIPTIONS[label]
            label_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]

            yield ImageTextSample(
                image=img_arr, caption=caption,
                source="cifar10",
                label=label_names[label] if label < 10 else "",
                metadata={"label_id": label})
            count += 1
            if count >= limit:
                return
    except Exception as e:
        logger.warning("CIFAR-10 fetch failed: %s", e)


def fetch_image_descriptions(limit: int = 500) -> Iterator[ImageTextSample]:
    """Fetch image-caption pairs from food101 (simpler than Flickr)."""
    if not _HF_AVAILABLE:
        return
    try:
        ds = _load_dataset_with_timeout("ethz/food101", split="train",
                          streaming=True, trust_remote_code=True)
        count = 0
        for item in _iter_with_stall_guard(ds):
            label = item.get('label', -1)
            img = item.get('image', None)

            if img is None:
                continue

            if hasattr(img, 'convert'):
                img_arr = np.array(img.convert('RGB'))
            else:
                continue

            # Resize to reasonable size (max 64x64 for efficiency)
            if img_arr.shape[0] > 64 or img_arr.shape[1] > 64:
                # Simple downsample
                h, w = img_arr.shape[:2]
                step_h = max(1, h // 64)
                step_w = max(1, w // 64)
                img_arr = img_arr[::step_h, ::step_w]

            # food101 labels are integers — map to name
            # (The dataset has 101 food categories)
            caption = f"A photograph of food (category {label})."

            yield ImageTextSample(
                image=img_arr, caption=caption,
                source="food101", label=str(label))
            count += 1
            if count >= limit:
                return
    except Exception as e:
        logger.warning("Food101 fetch failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# Built-in Rich Knowledge Corpus
# ═══════════════════════════════════════════════════════════════════════════

_RICH_CORPUS = [
    # ── Detailed factual knowledge ──
    "The human heart beats approximately 100,000 times per day, pumping about 2,000 gallons of blood through 60,000 miles of blood vessels.",
    "Mount Everest is the tallest mountain on Earth, standing at 8,849 metres above sea level. It is located in the Himalayas on the border of Nepal and Tibet.",
    "The Great Wall of China stretches over 13,000 miles. It was built over many centuries to protect Chinese states from invasions.",
    "The Amazon Rainforest produces about 20 percent of the world's oxygen and contains approximately 10 percent of all species on Earth.",
    "The Pacific Ocean is the largest ocean on Earth, covering about 63 million square miles. It contains more than half of the free water on Earth.",
    "Albert Einstein developed the theory of relativity, which fundamentally changed our understanding of space, time, and energy. His famous equation E equals mc squared describes the relationship between mass and energy.",
    "Photosynthesis is the process by which green plants convert carbon dioxide and water into glucose and oxygen using sunlight. This process is essential for life on Earth.",
    "The periodic table organizes chemical elements by their atomic number. Hydrogen is the lightest element with atomic number 1, while oganesson is the heaviest known element.",
    "Democracy originated in ancient Athens around 500 BCE. In a democracy, citizens have the right to vote and participate in government decisions.",
    "The internet was developed from ARPANET in the 1960s. Tim Berners-Lee invented the World Wide Web in 1989, making the internet accessible to the general public.",
    # ── Additional factual knowledge ──
    "Paris is the capital of France. France is a country in Western Europe known for its history, culture, and cuisine.",
    "Mercury is the closest planet to the Sun and the smallest planet in our solar system. Venus is the second closest, followed by Earth and Mars.",
    "The chemical symbol for water is H2O, meaning each molecule contains two hydrogen atoms and one oxygen atom. Water covers about 71 percent of the Earth's surface.",
    "There are 7 continents on Earth: Africa, Antarctica, Asia, Australia, Europe, North America, and South America. Asia is the largest continent by both area and population.",
    "The speed of light is approximately 299,792 kilometres per second. Nothing can travel faster than the speed of light in a vacuum.",

    # ── Conversational knowledge ──
    "When someone says 'hello' or asks 'how are you', they are making a social greeting. A common response is 'I am fine, thank you' followed by asking the same question back.",
    "When someone says 'thank you' or 'thanks', the polite response is 'you are welcome' or 'glad I could help'. Expressing gratitude is an important social skill.",
    "A good conversation involves listening to the other person, asking follow-up questions, and sharing relevant information or experiences.",
    "When explaining something complex, it helps to start with simple concepts and build up to more difficult ideas. Using examples makes explanations clearer.",
    "To answer a question well, first understand what is being asked, then provide relevant information, and finally check if the answer makes sense.",
    "Clear writing uses short sentences, specific words, and a logical structure. Each paragraph should focus on one main idea.",

    # ── Reasoning patterns ──
    "If it rains and the ground is flat, then puddles will form. This is because water accumulates in areas where it cannot drain away.",
    "All mammals breathe air. Whales are mammals. Therefore, whales breathe air, even though they live in the ocean.",
    "Correlation does not imply causation. Just because two events occur together does not mean one caused the other.",
    "When solving a problem, it helps to break it into smaller parts. Solve each part separately, then combine the solutions.",
    "Analogies help us understand new concepts by comparing them to familiar ones. For example, an atom is like a tiny solar system with electrons orbiting the nucleus.",

    # ── Math concepts ──
    "Addition combines two numbers to get a sum. For example, 3 plus 5 equals 8. The order does not matter: 5 plus 3 also equals 8.",
    "Subtraction finds the difference between two numbers. For example, 10 minus 4 equals 6. Unlike addition, order matters in subtraction.",
    "Multiplication is repeated addition. For example, 4 times 3 means adding 4 three times: 4 plus 4 plus 4 equals 12.",
    "Division splits a number into equal parts. For example, 12 divided by 3 equals 4, because 12 can be split into 3 groups of 4.",
    "Fractions represent parts of a whole. One half means 1 out of 2 equal parts. Three quarters means 3 out of 4 equal parts.",
    "Percentages express a number as parts per hundred. 50 percent means 50 out of 100, which is the same as one half.",
    "The area of a rectangle is calculated by multiplying its length by its width. A rectangle with length 5 and width 3 has an area of 15 square units.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",

    # ── Science explanations ──
    "The water cycle describes how water moves through the environment. Water evaporates from oceans, forms clouds, falls as rain, flows into rivers, and returns to the ocean.",
    "Cells are the basic units of life. Plant cells have cell walls and chloroplasts, while animal cells do not. Both types have a nucleus containing DNA.",
    "Newton's three laws of motion describe how objects move. The first law says objects at rest stay at rest unless acted on by a force. The second law says force equals mass times acceleration. The third law says every action has an equal and opposite reaction.",
    "Evolution by natural selection explains how species change over time. Organisms with traits better suited to their environment are more likely to survive and reproduce.",
    "The electromagnetic spectrum includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays, arranged by wavelength from longest to shortest.",

    # ── Technology ──
    "A computer program is a set of instructions that tells a computer what to do. Programs are written in programming languages like Python, Java, and C++.",
    "Machine learning allows computers to learn patterns from data without being explicitly programmed. It is used for image recognition, language translation, and recommendation systems.",
    "Encryption converts readable text into coded text that can only be decoded with a key. This protects sensitive information like passwords and financial data.",
    "A database is an organized collection of data stored electronically. Databases use structured query language (SQL) to retrieve and manipulate data.",
    "The cloud refers to servers and services accessed over the internet. Cloud computing allows users to store data and run applications without owning physical hardware.",

    # ── Geography and Culture ──
    "Africa is the second largest continent with 54 countries. Nigeria is the most populous country in Africa with over 200 million people.",
    "The Nile is the longest river in the world, flowing through northeastern Africa for about 6,650 kilometres.",
    "The Renaissance was a period of cultural and intellectual rebirth in Europe from the 14th to 17th centuries. It produced great art, science, and literature.",
    "The Industrial Revolution began in Britain in the late 18th century. It transformed society from agricultural to industrial, introducing factories and mass production.",
    "The United Nations was founded in 1945 to promote international cooperation and prevent future wars. It has 193 member states.",

    # ── Daily life and practical knowledge ──
    "A balanced diet includes fruits, vegetables, proteins, grains, and dairy. Eating a variety of foods provides the nutrients your body needs.",
    "Exercise strengthens muscles, improves heart health, reduces stress, and helps maintain a healthy weight. Adults should aim for at least 150 minutes of moderate exercise per week.",
    "Sleep is essential for physical and mental health. Most adults need 7 to 9 hours of sleep per night for optimal functioning.",
    "Reading improves vocabulary, comprehension, and critical thinking skills. Regular reading across different subjects broadens knowledge and perspective.",
    "Time management involves planning and prioritizing tasks. Breaking large projects into smaller steps and setting deadlines helps accomplish goals efficiently.",
]


def stream_rich_corpus() -> Iterator[TextSample]:
    """Stream the built-in rich knowledge corpus."""
    for text in _RICH_CORPUS:
        yield TextSample(text=text, source="rich_corpus",
                         category="knowledge")


# ═══════════════════════════════════════════════════════════════════════════
# Unified Fetcher
# ═══════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """Unified interface for fetching all dataset types.

    Usage::

        fetcher = DataFetcher()
        for sample in fetcher.fetch_text(limit=1000):
            engine.train_on_text(sample.text)
        for sample in fetcher.fetch_images(limit=100):
            engine.train_on_image(sample.image, sample.caption)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), ".data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._stats = Counter()

    def fetch_text(self, limit: int = 5000,
                   sources: Optional[List[str]] = None
                   ) -> Iterator[TextSample]:
        """Fetch text from all available sources.

        Args:
            limit: Maximum total samples.
            sources: Which sources to use. None = all available.
        """
        all_sources = sources or [
            "rich_corpus", "wikitext", "tinystories", "dolly"]

        per_source = max(100, limit // len(all_sources))
        total = 0

        for source in all_sources:
            if total >= limit:
                break
            source_count = 0
            try:
                gen = self._text_generator(source, per_source)
                for sample in gen:
                    if total >= limit:
                        break
                    yield sample
                    total += 1
                    source_count += 1
            except Exception as e:
                logger.warning("Source %s failed: %s", source, e)
            self._stats[f"text_{source}"] = source_count
            logger.info("Fetched %d text samples from %s",
                        source_count, source)

    def fetch_qa(self, limit: int = 1000,
                 sources: Optional[List[str]] = None
                 ) -> Iterator[QASample]:
        """Fetch Q&A pairs for training and evaluation."""
        all_sources = sources or ["openbookqa", "arc", "piqa"]
        per_source = max(100, limit // len(all_sources))
        total = 0

        for source in all_sources:
            if total >= limit:
                break
            source_count = 0
            gen = self._qa_generator(source, per_source)
            for sample in gen:
                if total >= limit:
                    break
                yield sample
                total += 1
                source_count += 1
            self._stats[f"qa_{source}"] = source_count

    def fetch_math(self, limit: int = 500) -> Iterator[MathSample]:
        """Fetch math problems."""
        count = 0
        for sample in fetch_gsm8k(limit):
            yield sample
            count += 1
        self._stats["math_gsm8k"] = count

    def fetch_images(self, limit: int = 500,
                     sources: Optional[List[str]] = None
                     ) -> Iterator[ImageTextSample]:
        """Fetch image-text pairs."""
        all_sources = sources or ["cifar10"]
        per_source = max(50, limit // len(all_sources))
        total = 0

        for source in all_sources:
            if total >= limit:
                break
            source_count = 0
            gen = self._image_generator(source, per_source)
            for sample in gen:
                if total >= limit:
                    break
                yield sample
                total += 1
                source_count += 1
            self._stats[f"image_{source}"] = source_count

    def fetch_conversations(self, limit: int = 500
                            ) -> Iterator[ConversationSample]:
        """Fetch conversation pairs."""
        count = 0
        for sample in fetch_conversations(limit):
            yield sample
            count += 1
        self._stats["conversations"] = count

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ── Internal generators ──

    def _text_generator(self, source: str, limit: int):
        if source == "rich_corpus":
            yield from stream_rich_corpus()
        elif source == "wikitext":
            yield from fetch_wikitext(limit)
        elif source == "tinystories":
            yield from fetch_tinystories(limit)
        elif source == "dolly":
            yield from fetch_dolly_conversations(limit)
        else:
            logger.warning("Unknown text source: %s", source)

    def _qa_generator(self, source: str, limit: int):
        if source == "openbookqa":
            yield from fetch_openbookqa(limit)
        elif source == "arc":
            yield from fetch_arc(limit)
        elif source == "piqa":
            yield from fetch_piqa(limit)
        else:
            logger.warning("Unknown QA source: %s", source)

    def _image_generator(self, source: str, limit: int):
        if source == "cifar10":
            yield from fetch_cifar10_pairs(limit)
        elif source == "food101":
            yield from fetch_image_descriptions(limit)
        else:
            logger.warning("Unknown image source: %s", source)
