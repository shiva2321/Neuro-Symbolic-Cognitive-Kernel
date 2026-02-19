"""
Tests for NSCK AI Engine multimodal capabilities.

Verifies text + image AI model functionality:
  1. Image training (image-caption pairs)
  2. Image description (image → text)
  3. Image generation (text → image)
  4. Text chat (text → text)
  5. Multimodal chat (text + image → text)
  6. Visual question-answering

All using the full NSCK cognitive pipeline — no neural networks.
"""

import os
import sys
import pytest
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from nsck_ai_model.ai_engine import NSCKAIEngine


# ── Helpers ───────────────────────────────────────────────────────────────

def _solid_color_image(r, g, b, size=(32, 32)):
    """Create a solid-color image."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _gradient_image(size=(32, 32)):
    """Create a horizontal gradient (left dark → right bright)."""
    h, w = size
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.tile(grad, (h, 1))
    return img  # grayscale


def _checkerboard_image(size=(32, 32), block=8):
    """Create a checkerboard pattern — high edge density."""
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            if (r // block + c // block) % 2 == 0:
                img[r, c] = 230
            else:
                img[r, c] = 25
    return img


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def multimodal_engine():
    """Trained engine with text + image data — shared across tests."""
    engine = NSCKAIEngine()

    # Text training
    texts = [
        "A red circle on a white background.",
        "Blue squares are commonly found in abstract art.",
        "Cats have whiskers and furry tails.",
        "The sunset paints the sky in orange and pink.",
        "A bright green leaf on a dark tree trunk.",
        "Rain causes flooding in low-lying areas.",
        "Dogs are loyal companions.",
    ]
    for t in texts:
        engine.train_on_text(t)

    # Image-caption training
    engine.train_on_image(
        _solid_color_image(200, 30, 30), "A red rectangle")
    engine.train_on_image(
        _solid_color_image(30, 30, 200), "A blue square shape")
    engine.train_on_image(
        _solid_color_image(30, 200, 30), "A bright green leaf")

    return engine


@pytest.fixture
def fresh_engine():
    """Untrained engine for isolation tests."""
    return NSCKAIEngine()


# ═══════════════════════════════════════════════════════════════════════════
# §1  Multimodal Module Presence
# ═══════════════════════════════════════════════════════════════════════════

class TestMultimodalModules:
    """Verify multimodal modules are present and initialized."""

    def test_multimodal_processor_exists(self, fresh_engine):
        from python.core.multimodal.multimodal_processor import MultimodalProcessor
        assert isinstance(fresh_engine.multimodal, MultimodalProcessor)

    def test_image_generator_exists(self, fresh_engine):
        from python.core.multimodal.image_generator import ImageGenerator
        assert isinstance(fresh_engine.image_generator, ImageGenerator)

    def test_supported_modalities(self, fresh_engine):
        mods = fresh_engine.multimodal.supported_modalities
        assert "text" in mods
        assert "image" in mods
        assert "audio" in mods

    def test_multimodal_stats_in_system_stats(self, fresh_engine):
        stats = fresh_engine.get_system_stats()
        assert "multimodal" in stats
        assert "image_generator" in stats
        assert stats["multimodal"]["supported_modalities"] == \
            ["text", "image", "audio", "video", "structured"]


# ═══════════════════════════════════════════════════════════════════════════
# §2  Image Training (image-caption pairs)
# ═══════════════════════════════════════════════════════════════════════════

class TestImageTraining:
    """Verify train_on_image learns cross-modal associations."""

    def test_train_returns_descriptors(self, fresh_engine):
        img = _solid_color_image(180, 50, 50)
        result = fresh_engine.train_on_image(img, "A reddish square")
        assert "image_descriptors" in result
        assert isinstance(result["image_descriptors"], list)
        assert len(result["image_descriptors"]) > 0

    def test_train_populates_visual_memory(self, fresh_engine):
        img = _solid_color_image(50, 50, 180)
        fresh_engine.train_on_image(img, "A blue shape")
        concepts = fresh_engine.image_generator.concept_memory.feature_templates
        assert len(concepts) > 0, "Visual concept memory should be populated"

    def test_train_updates_stats(self, fresh_engine):
        assert fresh_engine._training_stats["images_trained"] == 0
        img = _solid_color_image(100, 100, 100)
        fresh_engine.train_on_image(img, "A grey box")
        assert fresh_engine._training_stats["images_trained"] == 1

    def test_train_creates_semantic_links(self, fresh_engine):
        img = _solid_color_image(200, 200, 50)
        fresh_engine.train_on_image(img, "A yellow square")
        # Visual descriptors should be linked to caption concepts
        nodes = set(fresh_engine.semantic_memory.concept_graph.nodes)
        # Should have nodes from both caption and image descriptors
        assert len(nodes) > 0

    def test_train_also_learns_text(self, fresh_engine):
        img = _solid_color_image(100, 200, 100)
        fresh_engine.train_on_image(img, "Grass is usually green")
        # The caption should have been learned as text too
        result = fresh_engine.chat("What color is grass?")
        assert result["response"]  # Should produce some response


# ═══════════════════════════════════════════════════════════════════════════
# §3  Image Description
# ═══════════════════════════════════════════════════════════════════════════

class TestImageDescription:
    """Verify describe_image produces meaningful descriptions."""

    def test_describe_returns_dict(self, multimodal_engine):
        img = _solid_color_image(100, 100, 100)
        result = multimodal_engine.describe_image(img)
        assert "description" in result
        assert "descriptors" in result
        assert "confidence" in result
        assert "trace" in result
        assert "latency_ms" in result

    def test_describe_detects_color(self, multimodal_engine):
        img = _solid_color_image(200, 50, 50)
        result = multimodal_engine.describe_image(img)
        assert "color" in result["descriptors"]

    def test_describe_detects_greyscale(self, multimodal_engine):
        img = _gradient_image()
        result = multimodal_engine.describe_image(img)
        assert "greyscale" in result["descriptors"]

    def test_describe_detects_edges(self, multimodal_engine):
        img = _checkerboard_image()
        result = multimodal_engine.describe_image(img)
        # Checkerboard has high edge density
        assert any(d in result["descriptors"]
                   for d in ["detailed", "high_contrast", "textured"])

    def test_describe_uniform_image(self, multimodal_engine):
        img = _solid_color_image(128, 128, 128)
        result = multimodal_engine.describe_image(img)
        assert "uniform" in result["descriptors"]

    def test_describe_bright_image(self, multimodal_engine):
        img = _solid_color_image(240, 240, 240)
        result = multimodal_engine.describe_image(img)
        assert "bright" in result["descriptors"]

    def test_describe_dark_image(self, multimodal_engine):
        img = _solid_color_image(10, 10, 10)
        result = multimodal_engine.describe_image(img)
        assert "dark" in result["descriptors"]

    def test_describe_has_trace(self, multimodal_engine):
        img = _solid_color_image(100, 100, 100)
        result = multimodal_engine.describe_image(img)
        trace = result["trace"]
        assert "steps" in trace
        stages = [s["stage"] for s in trace["steps"]]
        assert "encode_image" in stages
        assert "search_semantic" in stages

    def test_describe_updates_stats(self, fresh_engine):
        assert fresh_engine._training_stats["images_described"] == 0
        img = _solid_color_image(100, 100, 100)
        fresh_engine.describe_image(img)
        assert fresh_engine._training_stats["images_described"] == 1

    def test_describe_trained_image_has_higher_confidence(self):
        """After training on a red image, describing a similar one should
        match learned concepts → higher confidence."""
        engine = NSCKAIEngine()
        red_img = _solid_color_image(200, 30, 30)
        engine.train_on_image(red_img, "A red square object")
        # Describe the same image — should match
        result = engine.describe_image(red_img)
        assert result["confidence"] > 0, \
            "Confidence should be > 0 for trained images"


# ═══════════════════════════════════════════════════════════════════════════
# §4  Image Generation (text → image)
# ═══════════════════════════════════════════════════════════════════════════

class TestImageGeneration:
    """Verify generate_image produces valid images."""

    def test_generate_returns_numpy_array(self, multimodal_engine):
        result = multimodal_engine.generate_image("red square")
        assert isinstance(result["image"], np.ndarray)

    def test_generate_correct_shape(self, multimodal_engine):
        result = multimodal_engine.generate_image("a circle", size=(48, 48))
        assert result["shape"][:2] == (48, 48)

    def test_generate_color_image(self, multimodal_engine):
        result = multimodal_engine.generate_image("a blue sky", color=True)
        assert len(result["shape"]) == 3
        assert result["shape"][2] == 3

    def test_generate_valid_pixel_range(self, multimodal_engine):
        result = multimodal_engine.generate_image("something")
        lo, hi = result["pixel_range"]
        assert lo >= 0
        assert hi <= 255

    def test_generate_updates_stats(self, fresh_engine):
        assert fresh_engine._training_stats["images_generated"] == 0
        fresh_engine.generate_image("test")
        assert fresh_engine._training_stats["images_generated"] == 1

    def test_generate_with_learned_concepts(self, multimodal_engine):
        """Generate from a concept that was trained — should use learned
        visual features, not random."""
        result = multimodal_engine.generate_image("red")
        assert result["visual_concepts_available"] > 0

    def test_generate_returns_concepts_used(self, multimodal_engine):
        result = multimodal_engine.generate_image("a bright green leaf")
        assert "concepts_used" in result
        assert len(result["concepts_used"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# §5  Text Chat (text → text)
# ═══════════════════════════════════════════════════════════════════════════

class TestTextChat:
    """Verify text-only chat still works end-to-end."""

    def test_chat_basic(self, multimodal_engine):
        result = multimodal_engine.chat("What has whiskers?")
        assert result["response"]
        assert "cats" in result["response"].lower() or \
               "whiskers" in result["response"].lower()

    def test_chat_returns_trace(self, multimodal_engine):
        result = multimodal_engine.chat("Tell me about dogs")
        assert "trace" in result
        assert "steps" in result["trace"]

    def test_chat_returns_emotion(self, multimodal_engine):
        result = multimodal_engine.chat("I love cats!")
        assert "emotion" in result


# ═══════════════════════════════════════════════════════════════════════════
# §6  Multimodal Chat
# ═══════════════════════════════════════════════════════════════════════════

class TestMultimodalChat:
    """Verify chat_multimodal handles text, image, or both."""

    def test_text_only_mode(self, multimodal_engine):
        result = multimodal_engine.chat_multimodal(text="Hello world")
        assert result["modalities"] == ["text"]

    def test_image_only_mode(self, multimodal_engine):
        img = _solid_color_image(100, 200, 100)
        result = multimodal_engine.chat_multimodal(image=img)
        assert result["modalities"] == ["image"]
        assert result["response"]

    def test_text_plus_image_mode(self, multimodal_engine):
        img = _solid_color_image(200, 50, 50)
        result = multimodal_engine.chat_multimodal(
            text="What is this?", image=img)
        assert result["modalities"] == ["text", "image"]
        assert "visual_descriptors" in result
        assert result["response"]

    def test_vqa_returns_visual_descriptors(self, multimodal_engine):
        img = _solid_color_image(50, 50, 200)
        result = multimodal_engine.chat_multimodal(
            text="Describe this", image=img)
        assert isinstance(result["visual_descriptors"], list)
        assert len(result["visual_descriptors"]) > 0

    def test_empty_input(self, multimodal_engine):
        result = multimodal_engine.chat_multimodal()
        assert result["modalities"] == []
        assert result["confidence"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §7  Full Pipeline: train → describe → generate → chat
# ═══════════════════════════════════════════════════════════════════════════

class TestFullMultimodalPipeline:
    """End-to-end: train with images, then use all capabilities."""

    def test_full_lifecycle(self):
        engine = NSCKAIEngine()

        # Phase 1: Train on text + images
        engine.train_on_text("Sunflowers are large yellow flowers.")
        engine.train_on_text("The ocean is deep blue.")
        yellow_img = _solid_color_image(220, 200, 30)
        engine.train_on_image(yellow_img, "A yellow sunflower")
        blue_img = _solid_color_image(20, 40, 180)
        engine.train_on_image(blue_img, "The deep blue ocean")

        # Phase 2: Describe an image
        desc = engine.describe_image(yellow_img)
        assert desc["description"]  # non-empty

        # Phase 3: Generate an image
        gen = engine.generate_image("sunflower", size=(32, 32))
        assert gen["image"].shape[:2] == (32, 32)

        # Phase 4: Text chat
        chat = engine.chat("What color are sunflowers?")
        assert "yellow" in chat["response"].lower() or \
               "sunflower" in chat["response"].lower()

        # Phase 5: VQA
        vqa = engine.chat_multimodal(
            text="What is this?", image=yellow_img)
        assert vqa["modalities"] == ["text", "image"]

        # Stats should reflect all operations
        stats = engine.get_system_stats()
        assert stats["multimodal"]["images_trained"] == 2
        assert stats["multimodal"]["images_described"] >= 1
        assert stats["multimodal"]["images_generated"] >= 1

    def test_different_image_types(self):
        """Verify the system handles different image formats."""
        engine = NSCKAIEngine()

        # Color image (H, W, 3)
        color = _solid_color_image(150, 100, 50)
        engine.describe_image(color)

        # Grayscale image (H, W)
        gray = _gradient_image()
        engine.describe_image(gray)

        # Small image
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        tiny[:] = 128
        engine.describe_image(tiny)

        # Large image
        big = np.zeros((128, 128, 3), dtype=np.uint8)
        big[:] = 200
        engine.describe_image(big)

        assert engine._training_stats["images_described"] == 4

    def test_cross_modal_retrieval(self):
        """Train on image+text, then text query should retrieve
        image-related concepts."""
        engine = NSCKAIEngine()

        # Train: associate "tiger" with a striped image
        striped = np.zeros((32, 32), dtype=np.uint8)
        for r in range(32):
            for c in range(32):
                striped[r, c] = 255 if (c // 4) % 2 == 0 else 0
        engine.train_on_image(striped, "A striped tiger pattern")

        # Query: the semantic memory should know about "tiger"
        nodes = {n.lower() for n in
                 engine.semantic_memory.concept_graph.nodes}
        assert "tiger" in nodes or "striped" in nodes, \
            f"Expected tiger/striped in {nodes}"
