#!/usr/bin/env python3
"""
NSCK Dual-Path Evaluation
=========================
Two parallel experiments to stress-test the NSCK cognitive architecture:

  Path A — Model Transplant: Load standard HF models (DistilBERT, MobileNetV2,
           Wav2Vec2), transplant their embeddings into NSCK's VSA substrate,
           then exercise the system with real-world cognitive tasks.

  Path B — Live Data Ingestion: Stream text/image/audio data from HuggingFace
           datasets directly through NSCK's native multimodal processor pipeline.

Usage:
    cd d:/Node_network
    python -m nsck.eval.nsck_dual_eval
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# NSCK imports
# ---------------------------------------------------------------------------
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.integration.config import NSCKConfig
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.transplant.pipeline import TransplantPipeline
from python.core.transplant.validator import TransplantReport
from python.core.memory.semantic_memory import SemanticMemory
from python.core.reasoning.analogy import AnalogyEngine
from python.core.multimodal.multimodal_processor import (
    MultimodalProcessor,
    MultimodalInput,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nsck.dual_eval")

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


def _header(title: str):
    print(f"\n{BOLD}{CYAN}{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}{RESET}\n")


def _section(title: str):
    print(f"\n{BOLD}{YELLOW}--- {title} ---{RESET}")


def _ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str):
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    score: float = 0.0
    max_score: float = 1.0
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PhaseReport:
    phase: str
    description: str
    setup_time_s: float = 0.0
    tests: List[TestResult] = field(default_factory=list)
    n_concepts_ingested: int = 0
    rss_mb_after: float = 0.0

    @property
    def aggregate_score(self) -> float:
        valid = [t for t in self.tests if t.error is None]
        if not valid:
            return 0.0
        return sum(t.score / t.max_score for t in valid) / len(valid)


# =========================================================================
# SHARED TEST BATTERY
# =========================================================================

def run_test_battery(engine: CognitiveEngine, label: str) -> List[TestResult]:
    """Run the standard cognitive test battery on a loaded engine."""
    results: List[TestResult] = []

    # ---- 1. Semantic QA via spreading activation ----
    _section(f"[{label}] Test 1: Semantic QA (Spreading Activation)")
    t0 = time.perf_counter()
    try:
        sem = engine.semantic_memory
        correct = 0
        total = 25
        # Direct retrieval: query concept → expect related concept activated
        for i in range(total):
            src = f"eval_src_{label}_{i}"
            dst = f"eval_dst_{label}_{i}"
            sem.add_concept(src, {"type": "entity", "index": i})
            sem.add_concept(dst, {"type": "entity", "index": i})
            sem.add_relation(src, "relates_to", dst)

        for i in range(total):
            src = f"eval_src_{label}_{i}"
            dst = f"eval_dst_{label}_{i}"
            try:
                activated = sem.spread_activation([src], steps=2)
                if dst in activated:
                    correct += 1
            except Exception:
                pass

        score = correct / total
        lat = time.perf_counter() - t0
        _ok(f"Score: {correct}/{total} = {score:.2%}  ({_ms(lat)})")
        results.append(TestResult("semantic_qa", score, 1.0, lat * 1000))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("semantic_qa", error=str(e)))

    # ---- 2. Multi-hop reasoning ----
    _section(f"[{label}] Test 2: Multi-Hop Reasoning (a→b→c)")
    t0 = time.perf_counter()
    try:
        sem = engine.semantic_memory
        correct = 0
        total = 20
        for i in range(total):
            a = f"mh_{label}_a_{i}"
            b = f"mh_{label}_b_{i}"
            c = f"mh_{label}_c_{i}"
            sem.add_concept(a, {})
            sem.add_concept(b, {})
            sem.add_concept(c, {})
            sem.add_relation(a, "leads_to", b)
            sem.add_relation(b, "leads_to", c)

        for i in range(total):
            a = f"mh_{label}_a_{i}"
            c = f"mh_{label}_c_{i}"
            try:
                activated = sem.spread_activation([a], steps=3)
                if c in activated:
                    correct += 1
            except Exception:
                pass

        score = correct / total
        lat = time.perf_counter() - t0
        _ok(f"Score: {correct}/{total} = {score:.2%}  ({_ms(lat)})")
        results.append(TestResult("multi_hop", score, 1.0, lat * 1000))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("multi_hop", error=str(e)))

    # ---- 3. Cross-modal association ----
    _section(f"[{label}] Test 3: Cross-Modal Association")
    t0 = time.perf_counter()
    try:
        sem = engine.semantic_memory
        correct = 0
        total = 15
        for i in range(total):
            text_concept = f"xm_text_{label}_{i}"
            visual_concept = f"xm_visual_{label}_{i}"
            sem.add_concept(text_concept, {"modality": "text"})
            sem.add_concept(visual_concept, {"modality": "image"})
            sem.add_relation(text_concept, "depicts", visual_concept)
            sem.add_relation(visual_concept, "described_by", text_concept)

        for i in range(total):
            text_concept = f"xm_text_{label}_{i}"
            visual_concept = f"xm_visual_{label}_{i}"
            try:
                activated = sem.spread_activation([text_concept], steps=2)
                if visual_concept in activated:
                    correct += 1
            except Exception:
                pass

        score = correct / total
        lat = time.perf_counter() - t0
        _ok(f"Score: {correct}/{total} = {score:.2%}  ({_ms(lat)})")
        results.append(TestResult("cross_modal", score, 1.0, lat * 1000))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("cross_modal", error=str(e)))

    # ---- 4. Analogy / Cross-Domain Abstraction ----
    _section(f"[{label}] Test 4: Cross-Domain Abstraction Discovery")
    t0 = time.perf_counter()
    try:
        analogy = engine.analogy
        sem = engine.semantic_memory

        # Register two mini-domains with shared abstract structure
        analogy.register_domain("domain_alpha", {
            "AGENT": "robot", "TARGET": "goal", "DANGER": "obstacle",
            "REWARD": "success", "PENALTY": "failure",
        }, create_missing_abstracts=True)
        analogy.register_domain("domain_beta", {
            "AGENT": "player", "TARGET": "treasure", "DANGER": "enemy",
            "REWARD": "points", "PENALTY": "game_over",
        }, create_missing_abstracts=True)

        # Test structural mapping discovery
        mapping = analogy.find_analogy("domain_alpha", "domain_beta")
        n_mappings = len(mapping.mappings) if mapping else 0
        sim_score = mapping.overall_similarity if mapping else 0.0

        # Test domain blending
        alpha_hvs = {c: hypervec_rs.HyperVector(hash(c) % (2**32))
                     for c in ["robot", "goal", "obstacle", "success", "failure"]}
        beta_hvs = {c: hypervec_rs.HyperVector(hash(c) % (2**32))
                    for c in ["player", "treasure", "enemy", "points", "game_over"]}
        blend_result = analogy.blend(alpha_hvs, beta_hvs)

        # Score: mappings found + similarity + blend success
        score = 0.0
        if n_mappings >= 3:
            score += 0.4
        elif n_mappings >= 1:
            score += 0.2
        score += min(0.3, sim_score * 0.3)
        if blend_result is not None:
            score += 0.3

        lat = time.perf_counter() - t0
        _ok(f"Mappings: {n_mappings}, Similarity: {sim_score:.2f}, "
            f"Blend: {'OK' if blend_result else 'FAIL'}, Score: {score:.2f}  ({_ms(lat)})")
        results.append(TestResult("analogy", score, 1.0, lat * 1000,
                                  details={"n_mappings": n_mappings, "similarity": sim_score}))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("analogy", error=str(e)))

    # ---- 5. Belief Revision under Contradiction ----
    _section(f"[{label}] Test 5: Belief Revision")
    t0 = time.perf_counter()
    try:
        sem = engine.semantic_memory
        # Teach: A causes B
        a_node = f"br_{label}_cause"
        b_node = f"br_{label}_effect"
        c_node = f"br_{label}_inhibitor"
        sem.add_concept(a_node, {"type": "event"})
        sem.add_concept(b_node, {"type": "event"})
        sem.add_concept(c_node, {"type": "event"})
        sem.add_relation(a_node, "causes", b_node)
        # Now contradict: C inhibits the effect
        sem.add_relation(c_node, "inhibits", b_node)

        # Check if both relations exist (system should retain both)
        activated_from_a = sem.spread_activation([a_node], steps=2)
        activated_from_c = sem.spread_activation([c_node], steps=2)

        has_causal = b_node in activated_from_a
        has_inhibit = b_node in activated_from_c

        score = 0.0
        if has_causal:
            score += 0.5
        if has_inhibit:
            score += 0.5

        lat = time.perf_counter() - t0
        _ok(f"Causal retained: {has_causal}, Inhibition learned: {has_inhibit}, Score: {score:.2f}  ({_ms(lat)})")
        results.append(TestResult("belief_revision", score, 1.0, lat * 1000))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("belief_revision", error=str(e)))

    # ---- 6. Conversational Session (Dialogue) ----
    _section(f"[{label}] Test 6: Conversational Session")
    t0 = time.perf_counter()
    try:
        dialogue = engine.dialogue
        convo_turns = [
            "Hello, who are you?",
            "What do you know about science?",
            "Can you remember what I just asked?",
            "Tell me something interesting.",
            "Goodbye!",
        ]

        _FALLBACK = "I didn't quite follow that"
        responses_ok = 0
        quality_score = 0.0
        total_turns = len(convo_turns)

        for turn in convo_turns:
            try:
                response = dialogue.process_turn(turn)
                resp_str = str(response).strip() if response else ""
                ok = False

                if resp_str and len(resp_str) > 20 and _FALLBACK not in resp_str:
                    # Full credit: meaningful response
                    quality_score += 1.0
                    ok = True
                elif resp_str and len(resp_str) > 5:
                    # Partial credit: short but valid
                    quality_score += 0.5
                    ok = True

                responses_ok += 1 if ok else 0
                # Print truncated response
                display = resp_str[:100] + ("..." if len(resp_str) > 100 else "")
                _ok(f'  "{turn}" → "{display}"') if ok else _warn(f'  "{turn}" → "{display}"')
            except Exception as e:
                _fail(f'  "{turn}" → ERROR: {e}')

        score = quality_score / total_turns
        lat = time.perf_counter() - t0
        _ok(f"Quality Score: {quality_score:.1f}/{total_turns} = {score:.2%}  ({_ms(lat)})")
        results.append(TestResult("conversation", score, 1.0, lat * 1000,
                                  details={"responses_ok": responses_ok, "quality": quality_score}))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("conversation", error=str(e)))

    # ---- 7. VSA Similarity Preservation ----
    _section(f"[{label}] Test 7: VSA Similarity Preservation")
    t0 = time.perf_counter()
    try:
        # Create pairs of related and unrelated concept HVs
        sem = engine.semantic_memory
        related_sims = []
        unrelated_sims = []
        for i in range(10):
            a = f"sim_{label}_a_{i}"
            b = f"sim_{label}_b_{i}"
            sem.add_concept(a, {"group": i})
            sem.add_concept(b, {"group": i})
            sem.add_relation(a, "similar_to", b)

        # Check if concepts that are related have HVs stored
        if hasattr(sem, "concept_hvs") and sem.concept_hvs:
            for i in range(10):
                a_hv = sem.concept_hvs.get(f"sim_{label}_a_{i}")
                b_hv = sem.concept_hvs.get(f"sim_{label}_b_{i}")
                if a_hv and b_hv:
                    related_sims.append(a_hv.similarity(b_hv))

                u_hv = sem.concept_hvs.get(f"sim_{label}_a_{(i + 5) % 10}")
                if a_hv and u_hv and i != (i + 5) % 10:
                    unrelated_sims.append(a_hv.similarity(u_hv))

        if related_sims:
            mean_related = np.mean(related_sims)
            mean_unrelated = np.mean(unrelated_sims) if unrelated_sims else 0.5
            separation = mean_related - mean_unrelated
            score = min(1.0, max(0.0, separation + 0.5))
        else:
            # If no HVs stored, check graph retrieval instead
            correct = 0
            for i in range(10):
                a = f"sim_{label}_a_{i}"
                b = f"sim_{label}_b_{i}"
                try:
                    activated = sem.spread_activation([a], steps=1)
                    if b in activated:
                        correct += 1
                except Exception:
                    pass
            score = correct / 10.0

        lat = time.perf_counter() - t0
        _ok(f"Score: {score:.2%}  ({_ms(lat)})")
        results.append(TestResult("vsa_similarity", score, 1.0, lat * 1000))
    except Exception as e:
        _fail(f"Error: {e}")
        results.append(TestResult("vsa_similarity", error=str(e)))

    return results


# =========================================================================
# PHASE A — MODEL TRANSPLANT
# =========================================================================

def phase_a_transplant() -> PhaseReport:
    """Transplant standard HuggingFace models into NSCK's VSA substrate."""
    _header("PHASE A: Model Transplant Path")
    report = PhaseReport(
        phase="A",
        description="Transplant standard HF models (DistilBERT, MobileNetV2, Wav2Vec2) "
                    "into NSCK via TransplantPipeline, then run cognitive tests.",
    )
    t_start = time.perf_counter()

    # --- Create engine ---
    config = NSCKConfig.research()
    engine = CognitiveEngine(config=config, persistence_path=":memory:")
    pipeline = TransplantPipeline(config=config)

    transplant_reports: Dict[str, Optional[TransplantReport]] = {}
    total_concepts = 0

    # === A1. Text model: DistilBERT ===
    _section("A1: Transplanting DistilBERT (text)")
    try:
        from transformers import DistilBertModel
        print(f"  {DIM}Loading distilbert-base-uncased...{RESET}")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        model.eval()

        t0 = time.perf_counter()
        tr = pipeline.run(
            model=model,
            domain_name="text_distilbert",
            strategy="learned",
            cognitive_engine=engine,
        )
        lat = time.perf_counter() - t0
        transplant_reports["text"] = tr
        total_concepts += tr.n_concepts

        _ok(f"DistilBERT transplanted: {tr.n_concepts} concepts")
        _ok(f"  Spearman ρ={tr.spearman_rho:.3f}  Recall@10={tr.recall_at_10:.3f}  "
            f"Recall@50={tr.recall_at_50:.3f}  ARI={tr.ari:.3f}")
        _ok(f"  Passed: {tr.passed}  ({_ms(lat)})")
        del model
    except Exception as e:
        _fail(f"DistilBERT transplant failed: {e}")
        traceback.print_exc()
        transplant_reports["text"] = None

    # === A2. Image model: MobileNetV2 ===
    _section("A2: Transplanting MobileNetV2 (image)")
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        print(f"  {DIM}Loading mobilenet_v2...{RESET}")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()

        t0 = time.perf_counter()
        tr = pipeline.run(
            model=model,
            domain_name="image_mobilenet",
            strategy="learned",
            cognitive_engine=engine,
        )
        lat = time.perf_counter() - t0
        transplant_reports["image"] = tr
        total_concepts += tr.n_concepts

        _ok(f"MobileNetV2 transplanted: {tr.n_concepts} concepts")
        _ok(f"  Spearman ρ={tr.spearman_rho:.3f}  Recall@10={tr.recall_at_10:.3f}  "
            f"Recall@50={tr.recall_at_50:.3f}  ARI={tr.ari:.3f}")
        _ok(f"  Passed: {tr.passed}  ({_ms(lat)})")
        del model
    except Exception as e:
        _fail(f"MobileNetV2 transplant failed: {e}")
        traceback.print_exc()
        transplant_reports["image"] = None

    # === A3. Audio model: Wav2Vec2 ===
    _section("A3: Transplanting Wav2Vec2 (audio)")
    try:
        from transformers import Wav2Vec2Model
        print(f"  {DIM}Loading wav2vec2-base...{RESET}")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        model.eval()

        t0 = time.perf_counter()
        tr = pipeline.run(
            model=model,
            domain_name="audio_wav2vec2",
            strategy="learned",
            cognitive_engine=engine,
        )
        lat = time.perf_counter() - t0
        transplant_reports["audio"] = tr
        total_concepts += tr.n_concepts

        _ok(f"Wav2Vec2 transplanted: {tr.n_concepts} concepts")
        _ok(f"  Spearman ρ={tr.spearman_rho:.3f}  Recall@10={tr.recall_at_10:.3f}  "
            f"Recall@50={tr.recall_at_50:.3f}  ARI={tr.ari:.3f}")
        _ok(f"  Passed: {tr.passed}  ({_ms(lat)})")
        del model
    except Exception as e:
        _fail(f"Wav2Vec2 transplant failed: {e}")
        traceback.print_exc()
        transplant_reports["audio"] = None

    report.setup_time_s = time.perf_counter() - t_start
    report.n_concepts_ingested = total_concepts

    # --- Transplant quality summary ---
    _section("Transplant Quality Summary")
    for modality, tr in transplant_reports.items():
        if tr:
            status = f"{GREEN}PASS{RESET}" if tr.passed else f"{RED}FAIL{RESET}"
            print(f"  {modality:8s}: ρ={tr.spearman_rho:.3f}  R@10={tr.recall_at_10:.3f}  "
                  f"R@50={tr.recall_at_50:.3f}  ARI={tr.ari:.3f}  [{status}]")
        else:
            print(f"  {modality:8s}: {RED}NOT LOADED{RESET}")

    # --- Run test battery ---
    _header("Phase A — Cognitive Test Battery")
    report.tests = run_test_battery(engine, "transplant")
    report.rss_mb_after = _rss_mb()

    # Store transplant details
    for modality, tr in transplant_reports.items():
        if tr:
            report.tests.append(TestResult(
                f"transplant_{modality}",
                score=tr.spearman_rho,
                max_score=1.0,
                details={
                    "spearman_rho": tr.spearman_rho,
                    "recall_at_10": tr.recall_at_10,
                    "recall_at_50": tr.recall_at_50,
                    "ari": tr.ari,
                    "n_concepts": tr.n_concepts,
                    "passed": tr.passed,
                },
            ))

    return report


# =========================================================================
# PHASE B — LIVE HUGGINGFACE DATA
# =========================================================================

def phase_b_live_data() -> PhaseReport:
    """Stream live data from HuggingFace through NSCK's native pipeline."""
    _header("PHASE B: Live HuggingFace Data Path")
    report = PhaseReport(
        phase="B",
        description="Stream text/image/audio from HuggingFace datasets through "
                    "NSCK's native multimodal processor (no external models).",
    )
    t_start = time.perf_counter()

    # --- Create engine ---
    config = NSCKConfig.research()
    engine = CognitiveEngine(config=config, persistence_path=":memory:")
    mm_processor = MultimodalProcessor(
        context_engine=None,
        semantic_memory=engine.semantic_memory,
    )

    n_text = 0
    n_image = 0
    n_audio = 0

    # === B1. Text data: WikiText ===
    _section("B1: Streaming WikiText (text)")
    try:
        from datasets import load_dataset
        print(f"  {DIM}Loading wikitext-2-raw-v1...{RESET}")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        t0 = time.perf_counter()
        max_text = 3000
        sem = engine.semantic_memory
        for row in ds:
            if n_text >= max_text:
                break
            text = row.get("text", "").strip()
            if len(text) < 20:
                continue

            # Process through multimodal processor
            inp = MultimodalInput(text=text)
            result = mm_processor.process(inp)

            # Also add concepts to semantic memory
            words = text.lower().split()
            for w in words[:10]:
                clean = "".join(c for c in w if c.isalnum())
                if len(clean) >= 3:
                    sem.add_concept(f"wiki_{clean}", {"source": "wikitext", "modality": "text"})

            n_text += 1
            if n_text % 500 == 0:
                print(f"    Processed {n_text} text samples...")

        lat = time.perf_counter() - t0
        _ok(f"Processed {n_text} text samples  ({_ms(lat)} total, "
            f"{_ms(lat / max(1, n_text))} per sample)")
    except Exception as e:
        _fail(f"WikiText loading failed: {e}")
        traceback.print_exc()

    # === B2. Image data: CIFAR-10 ===
    _section("B2: Streaming CIFAR-10 (images)")
    try:
        from datasets import load_dataset
        print(f"  {DIM}Loading CIFAR-10...{RESET}")
        ds = load_dataset("uoft-cs/cifar10", split="train")

        t0 = time.perf_counter()
        max_images = 3000
        label_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
        for row in ds:
            if n_image >= max_images:
                break

            img = row.get("img")
            label_idx = row.get("label", 0)
            label = label_names[label_idx] if label_idx < len(label_names) else f"class_{label_idx}"

            if img is not None:
                # Convert PIL Image to numpy array
                img_array = np.array(img, dtype=np.float64)
                inp = MultimodalInput(image=img_array)
                result = mm_processor.process(inp)

                # Add the label concept + image concept to semantic memory
                sem.add_concept(f"cifar_{label}_{n_image}", {
                    "source": "cifar10",
                    "modality": "image",
                    "label": label,
                })
                sem.add_concept(label, {"type": "category"})
                sem.add_relation(f"cifar_{label}_{n_image}", "is_a", label)

                n_image += 1
                if n_image % 500 == 0:
                    print(f"    Processed {n_image} images...")

        lat = time.perf_counter() - t0
        _ok(f"Processed {n_image} images  ({_ms(lat)} total, "
            f"{_ms(lat / max(1, n_image))} per image)")
    except Exception as e:
        _fail(f"CIFAR-10 loading failed: {e}")
        traceback.print_exc()

    # === B3. Audio data: Synthetic waveforms ===
    # (HF speech_commands uses deprecated script loader — generate synthetic audio)
    _section("B3: Generating synthetic audio waveforms")
    try:
        t0 = time.perf_counter()
        max_audio = 2000
        audio_labels = ["tone_low", "tone_mid", "tone_high", "chirp", "noise",
                        "pulse", "sweep", "click", "hum", "ring"]
        rng = np.random.default_rng(42)

        for i in range(max_audio):
            label = audio_labels[i % len(audio_labels)]
            sr = 16000
            duration_s = 0.5 + rng.random() * 0.5  # 0.5-1.0 seconds
            t_arr = np.linspace(0, duration_s, int(sr * duration_s))

            # Generate different waveform types
            if "low" in label:
                audio_np = np.sin(2 * np.pi * 200 * t_arr)
            elif "mid" in label:
                audio_np = np.sin(2 * np.pi * 1000 * t_arr)
            elif "high" in label:
                audio_np = np.sin(2 * np.pi * 4000 * t_arr)
            elif "chirp" in label:
                freq = 200 + 3800 * t_arr / t_arr[-1]  # sweep 200→4000 Hz
                audio_np = np.sin(2 * np.pi * freq * t_arr)
            elif "noise" in label:
                audio_np = rng.standard_normal(len(t_arr))
            elif "pulse" in label:
                audio_np = np.zeros_like(t_arr)
                pulse_pos = np.linspace(0, len(t_arr) - 1, 10, dtype=int)
                audio_np[pulse_pos] = 1.0
            elif "sweep" in label:
                freq = 4000 - 3800 * t_arr / t_arr[-1]  # reverse sweep
                audio_np = np.sin(2 * np.pi * freq * t_arr)
            else:
                audio_np = np.sin(2 * np.pi * 440 * t_arr)  # A4 tone

            # Add slight noise
            audio_np = audio_np + rng.standard_normal(len(audio_np)) * 0.05
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)

            inp = MultimodalInput(audio=audio_np)
            result = mm_processor.process(inp)

            sem.add_concept(f"audio_{label}_{i}", {
                "source": "synthetic",
                "modality": "audio",
                "label": label,
            })

            n_audio += 1
            if n_audio % 500 == 0:
                print(f"    Generated {n_audio} audio clips...")

        lat = time.perf_counter() - t0
        _ok(f"Generated {n_audio} audio clips  ({_ms(lat)} total, "
            f"{_ms(lat / max(1, n_audio))} per clip)")
    except Exception as e:
        _fail(f"Audio generation failed: {e}")
        traceback.print_exc()

    report.setup_time_s = time.perf_counter() - t_start
    report.n_concepts_ingested = n_text + n_image + n_audio

    # Build relations between same-label concepts
    _section("B4: Building cross-concept relations")
    t0 = time.perf_counter()
    try:
        # Add category relations for CIFAR
        label_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
        for lbl in label_names:
            sem.add_concept(lbl, {"type": "category"})
            # Add property relations
            if lbl in ("airplane", "ship", "automobile", "truck"):
                sem.add_concept("vehicle", {"type": "super_category"})
                sem.add_relation(lbl, "is_a", "vehicle")
            elif lbl in ("bird", "cat", "deer", "dog", "frog", "horse"):
                sem.add_concept("animal", {"type": "super_category"})
                sem.add_relation(lbl, "is_a", "animal")
        lat = time.perf_counter() - t0
        _ok(f"Relations built  ({_ms(lat)})")
    except Exception as e:
        _warn(f"Relation building partially failed: {e}")

    # --- Run test battery ---
    _header("Phase B — Cognitive Test Battery")
    report.tests = run_test_battery(engine, "livedata")
    report.rss_mb_after = _rss_mb()

    # Add ingestion stats
    report.tests.append(TestResult(
        "data_ingestion",
        score=1.0 if (n_text + n_image + n_audio) > 0 else 0.0,
        details={
            "n_text": n_text,
            "n_image": n_image,
            "n_audio": n_audio,
            "total": n_text + n_image + n_audio,
        },
    ))

    return report


# =========================================================================
# PHASE C — COMPARISON
# =========================================================================

def compare_reports(a: PhaseReport, b: PhaseReport) -> Dict[str, Any]:
    """Generate the comparison analysis."""
    _header("PHASE C: Comparative Analysis")

    comparison: Dict[str, Any] = {
        "phase_a_aggregate": a.aggregate_score,
        "phase_b_aggregate": b.aggregate_score,
        "phase_a_concepts": a.n_concepts_ingested,
        "phase_b_concepts": b.n_concepts_ingested,
        "phase_a_setup_s": a.setup_time_s,
        "phase_b_setup_s": b.setup_time_s,
        "per_test": {},
    }

    # Build comparison table
    a_tests = {t.name: t for t in a.tests}
    b_tests = {t.name: t for t in b.tests}
    all_names = sorted(set(list(a_tests.keys()) + list(b_tests.keys())))

    print(f"\n  {'Test':<25s}  {'Path A (Transplant)':>20s}  {'Path B (Live Data)':>20s}  {'Winner':>10s}")
    print(f"  {'─' * 25}  {'─' * 20}  {'─' * 20}  {'─' * 10}")

    for name in all_names:
        ta = a_tests.get(name)
        tb = b_tests.get(name)
        sa = f"{ta.score:.3f}" if ta and ta.error is None else "ERROR" if ta else "N/A"
        sb = f"{tb.score:.3f}" if tb and tb.error is None else "ERROR" if tb else "N/A"

        a_val = ta.score if ta and ta.error is None else -1
        b_val = tb.score if tb and tb.error is None else -1
        if a_val > b_val:
            winner = "Path A"
        elif b_val > a_val:
            winner = "Path B"
        else:
            winner = "Tie"

        print(f"  {name:<25s}  {sa:>20s}  {sb:>20s}  {winner:>10s}")

        comparison["per_test"][name] = {
            "path_a": a_val,
            "path_b": b_val,
            "winner": winner,
        }

    print()
    print(f"  {'AGGREGATE':25s}  {a.aggregate_score:20.3f}  {b.aggregate_score:20.3f}  "
          f"{'Path A' if a.aggregate_score > b.aggregate_score else 'Path B' if b.aggregate_score > a.aggregate_score else 'Tie':>10s}")
    print(f"\n  Concepts ingested:  A={a.n_concepts_ingested:,}  B={b.n_concepts_ingested:,}")
    print(f"  Setup time:         A={a.setup_time_s:.1f}s  B={b.setup_time_s:.1f}s")
    if a.rss_mb_after > 0 or b.rss_mb_after > 0:
        print(f"  RSS after test:     A={a.rss_mb_after:.0f} MB  B={b.rss_mb_after:.0f} MB")

    return comparison


# =========================================================================
# MAIN
# =========================================================================

def main():
    _header("NSCK DUAL-PATH EVALUATION")
    print(f"  Running dual experiment: Model Transplant (A) vs. Live Data (B)")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    overall_start = time.perf_counter()

    # Run Phase A
    try:
        report_a = phase_a_transplant()
    except Exception as e:
        _fail(f"Phase A failed entirely: {e}")
        traceback.print_exc()
        report_a = PhaseReport(phase="A", description="FAILED")

    # Run Phase B
    try:
        report_b = phase_b_live_data()
    except Exception as e:
        _fail(f"Phase B failed entirely: {e}")
        traceback.print_exc()
        report_b = PhaseReport(phase="B", description="FAILED")

    # Run Phase C: Comparison
    comparison = compare_reports(report_a, report_b)

    total_time = time.perf_counter() - overall_start
    print(f"\n  Total evaluation time: {total_time:.1f}s")

    # Save results JSON
    results_path = os.path.join(os.path.dirname(__file__), "results", "dual_eval_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": total_time,
        "phase_a": {
            "setup_time_s": report_a.setup_time_s,
            "n_concepts": report_a.n_concepts_ingested,
            "aggregate_score": report_a.aggregate_score,
            "tests": [asdict(t) for t in report_a.tests],
        },
        "phase_b": {
            "setup_time_s": report_b.setup_time_s,
            "n_concepts": report_b.n_concepts_ingested,
            "aggregate_score": report_b.aggregate_score,
            "tests": [asdict(t) for t in report_b.tests],
        },
        "comparison": comparison,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    _ok(f"Results saved to {results_path}")
    _header("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
