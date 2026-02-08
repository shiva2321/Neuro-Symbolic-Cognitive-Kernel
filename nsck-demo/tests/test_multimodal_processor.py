"""
Test Multimodal Processor
Verifies: text, image, audio, structured data encoding and fusion.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from python.multimodal_processor import (
    MultimodalProcessor,
    MultimodalInput,
    ProcessedInput,
    ModalityResult,
)


def test_text_processing():
    """Test that text input is encoded into an HV with extracted tokens."""
    print("--- Test: Text Processing ---")

    proc = MultimodalProcessor()
    inp = MultimodalInput(text="The red rose is beautiful in the garden")
    result = proc.process(inp)

    assert len(result.modality_results) == 1
    assert result.modality_results[0].modality == "text"
    assert "red" in result.extracted_concepts
    assert "rose" in result.extracted_concepts
    assert result.confidence > 0.3
    print(f"  Tokens: {result.extracted_concepts}")
    print(f"  Confidence: {result.confidence:.2f}")

    print("✅ Text Processing Test Passed")


def test_image_processing():
    """Test that image input is encoded into an HV with descriptors."""
    print("\n--- Test: Image Processing ---")

    proc = MultimodalProcessor()
    # Create a bright test image (100x100, RGB)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 220
    inp = MultimodalInput(image=img)
    result = proc.process(inp)

    assert len(result.modality_results) == 1
    assert result.modality_results[0].modality == "image"
    stats = result.modality_results[0].features.get("stats", {})
    assert stats["height"] == 100
    assert stats["width"] == 100
    print(f"  Stats: {stats}")
    print(f"  Descriptors: {result.modality_results[0].features.get('descriptors', [])}")

    print("✅ Image Processing Test Passed")


def test_audio_processing():
    """Test that audio input is encoded into an HV."""
    print("\n--- Test: Audio Processing ---")

    proc = MultimodalProcessor()
    # Create a simple sine wave (1 second at 44100 Hz)
    t = np.linspace(0, 1, 44100, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    inp = MultimodalInput(audio=audio)
    result = proc.process(inp)

    assert len(result.modality_results) == 1
    assert result.modality_results[0].modality == "audio"
    stats = result.modality_results[0].features.get("stats", {})
    assert stats["length"] == 44100
    print(f"  Stats: {stats}")

    print("✅ Audio Processing Test Passed")


def test_structured_processing():
    """Test that structured dict is encoded into an HV."""
    print("\n--- Test: Structured Data Processing ---")

    proc = MultimodalProcessor()
    data = {"head": (5, 5), "food": (3, 7), "score": 10}
    inp = MultimodalInput(structured=data)
    result = proc.process(inp)

    assert len(result.modality_results) == 1
    assert result.modality_results[0].modality == "structured"
    predicates = result.modality_results[0].features.get("predicates", [])
    assert len(predicates) == 3
    print(f"  Predicates: {predicates}")

    print("✅ Structured Data Processing Test Passed")


def test_multimodal_fusion():
    """Test fusion of multiple modalities."""
    print("\n--- Test: Multimodal Fusion ---")

    proc = MultimodalProcessor()
    inp = MultimodalInput(
        text="A red rose in a garden",
        image=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        structured={"location": "garden", "type": "flower"},
    )
    result = proc.process(inp)

    assert len(result.modality_results) == 3
    modalities = {r.modality for r in result.modality_results}
    assert modalities == {"text", "image", "structured"}
    assert result.fused_hv is not None
    print(f"  Fused {len(result.modality_results)} modalities")
    print(f"  Concepts: {result.extracted_concepts}")

    print("✅ Multimodal Fusion Test Passed")


def test_empty_input():
    """Test that empty input is handled gracefully."""
    print("\n--- Test: Empty Input ---")

    proc = MultimodalProcessor()
    inp = MultimodalInput()
    result = proc.process(inp)

    assert result.fused_hv is not None
    assert result.confidence == 0.0
    assert len(result.modality_results) == 0
    print(f"  Empty input handled: confidence={result.confidence}")

    print("✅ Empty Input Test Passed")


def test_text_determinism():
    """Test that same text produces the same HV."""
    print("\n--- Test: Text Determinism ---")

    proc = MultimodalProcessor()
    text = "hello world"
    r1 = proc.process(MultimodalInput(text=text))
    r2 = proc.process(MultimodalInput(text=text))

    sim = r1.fused_hv.similarity(r2.fused_hv)
    assert sim > 0.99, f"Same text should produce near-identical HVs, got sim={sim}"
    print(f"  Similarity of identical text: {sim:.4f}")

    print("✅ Text Determinism Test Passed")


def test_different_texts_differ():
    """Test that different texts produce different HVs."""
    print("\n--- Test: Different Texts Differ ---")

    proc = MultimodalProcessor()
    r1 = proc.process(MultimodalInput(text="The cat sat on the mat"))
    r2 = proc.process(MultimodalInput(text="The dog ran in the park"))

    sim = r1.fused_hv.similarity(r2.fused_hv)
    assert sim < 0.9, f"Different texts should produce different HVs, got sim={sim}"
    print(f"  Similarity of different texts: {sim:.4f}")

    print("✅ Different Texts Differ Test Passed")


if __name__ == "__main__":
    test_text_processing()
    test_image_processing()
    test_audio_processing()
    test_structured_processing()
    test_multimodal_fusion()
    test_empty_input()
    test_text_determinism()
    test_different_texts_differ()
    print("\n🎉 All Multimodal Processor Tests Passed!")
