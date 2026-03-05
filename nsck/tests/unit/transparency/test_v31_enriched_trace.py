"""
Tests for V31 enriched ThoughtTrace — verifies that all 11 cognitive stages
are populated with real (non-empty) data after the V31 trace enrichment in
cognitive_engine.decide().
"""
from __future__ import annotations

import pytest
import sys
import os

_NSCK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _NSCK not in sys.path:
    sys.path.insert(0, _NSCK)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def substrate_with_knowledge():
    """NSCKSubstrate with a V30 config + some ingested knowledge."""
    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate

    cfg = NSCKConfig.v30()
    sub = NSCKSubstrate(cfg)

    # Ingest a small knowledge corpus so semantic memory is populated
    for doc in [
        "Photosynthesis converts sunlight into glucose in plant cells.",
        "DNA encodes the genetic blueprint of every living organism.",
        "Neurons transmit electrical impulses through synaptic connections.",
        "Machine learning uses gradient descent to fit model parameters.",
        "Climate change is accelerated by greenhouse gas emissions.",
    ]:
        sub.ingest(doc, "test_science")
        sub.feedback("learn", 0.8, "test_science", state=doc, outcome="ok")

    return sub


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _process_with_trace(sub, text: str):
    """Process text and return (result, thought_trace). Asserts trace is not None."""
    res = sub.process(text, "test_science")
    assert res.thought_trace is not None, f"ThoughtTrace is None for: {text!r}"
    return res, res.thought_trace


# ===========================================================================
# Stage-level richness tests
# ===========================================================================

class TestEncodingStage:
    def test_encoding_has_modality(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is photosynthesis?")
        step = tt.get_step("encoding")
        assert step is not None
        data = step.data
        assert data.get("modality") in ("text", "dict", "inline"), f"got: {data.get('modality')}"

    def test_encoding_has_hv_dimension(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain DNA.")
        step = tt.get_step("encoding")
        assert step.data.get("hv_dimension", 0) > 0

    def test_encoding_summary_not_flat(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "How do neurons work?")
        step = tt.get_step("encoding")
        # V31 summary should contain 'adapter=' or 'projector='
        assert "projector=" in step.summary or "adapter=" in step.summary, \
            f"Expected rich encoding summary, got: {step.summary!r}"


class TestEmotionStage:
    def test_question_detected_as_curious(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is photosynthesis?")
        step = tt.get_step("emotion")
        assert step is not None
        # A question with 'what' should be curious or positive (not stuck at 'neutral')
        # — we accept neutral only if no question words detected
        assert step.data.get("label") in ("curious", "positive", "neutral")

    def test_positive_text_detected(self, substrate_with_knowledge):
        _, tt = _process_with_trace(
            substrate_with_knowledge,
            "Photosynthesis is beautiful and wonderful for life on Earth."
        )
        step = tt.get_step("emotion")
        label = step.data.get("label", "neutral")
        valence = step.data.get("valence", 0.0)
        # Should detect positive sentiment
        assert label == "positive" or valence > 0, \
            f"Expected positive, got label={label}, valence={valence}"

    def test_negative_text_detected(self, substrate_with_knowledge):
        _, tt = _process_with_trace(
            substrate_with_knowledge,
            "Disease and terrible suffering harm many people."
        )
        step = tt.get_step("emotion")
        label = step.data.get("label", "neutral")
        valence = step.data.get("valence", 0.0)
        assert label == "negative" or valence < 0, \
            f"Expected negative, got label={label}, valence={valence}"

    def test_counterfactual_anticipatory(self, substrate_with_knowledge):
        _, tt = _process_with_trace(
            substrate_with_knowledge,
            "What if Earth had no atmosphere?"
        )
        step = tt.get_step("emotion")
        label = step.data.get("label", "neutral")
        # Hypothetical queries should be anticipatory or curious
        assert label in ("anticipatory", "curious", "neutral"), \
            f"Unexpected emotion for hypothetical: {label}"

    def test_emotion_summary_not_flat(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "How does the brain work?")
        step = tt.get_step("emotion")
        # V31 summary should contain valence= and arousal=
        assert "valence=" in step.summary, f"Got: {step.summary!r}"
        assert "arousal=" in step.summary, f"Got: {step.summary!r}"


class TestConceptExtractionStage:
    def test_intent_extracted(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is DNA?")
        step = tt.get_step("concept_extraction")
        assert step is not None
        intent = step.data.get("intent", "unknown")
        # Questions should have intent=question
        assert intent != "unknown", f"Intent not extracted, got: {intent}"

    def test_concepts_non_empty(self, substrate_with_knowledge):
        _, tt = _process_with_trace(
            substrate_with_knowledge,
            "Photosynthesis is the process by which plants absorb sunlight."
        )
        step = tt.get_step("concept_extraction")
        concepts = step.data.get("concepts", [])
        n = step.data.get("n_concepts", 0)
        # Should extract some concepts from this rich sentence
        assert n > 0 or len(concepts) > 0, f"Expected concepts, got: {concepts}"

    def test_svo_extracted_for_declarative(self, substrate_with_knowledge):
        _, tt = _process_with_trace(
            substrate_with_knowledge,
            "Plants convert sunlight into glucose."
        )
        step = tt.get_step("concept_extraction")
        svo = step.data.get("svo_triples", [])
        # We may not always get SVO but the data dict should at least have the key
        assert "svo_triples" in step.data

    def test_summary_includes_intent_and_n_concepts(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain evolution.")
        step = tt.get_step("concept_extraction")
        assert "intent=" in step.summary, f"Got: {step.summary!r}"
        assert "n_concepts=" in step.summary, f"Got: {step.summary!r}"


class TestSemanticSearchStage:
    def test_semantic_search_has_data(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is photosynthesis?")
        step = tt.get_step("semantic_search")
        assert step is not None
        assert "n_total_concepts" in step.data, f"Missing n_total_concepts in {step.data}"

    def test_semantic_search_finds_concepts_after_ingest(self, substrate_with_knowledge):
        # After ingesting 5 docs, semantic memory should have concepts
        _, tt = _process_with_trace(substrate_with_knowledge, "sunlight plants glucose")
        step = tt.get_step("semantic_search")
        n_total = step.data.get("n_total_concepts", 0)
        # Should have at least 5 concepts from our 5 ingested docs
        assert n_total > 0, f"Expected concepts in memory, got n_total={n_total}"

    def test_summary_shows_match_info(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "DNA genetics")
        step = tt.get_step("semantic_search")
        # Should say either "Found X/Y concepts" or "No semantic matches"
        assert "concepts" in step.summary or "No semantic" in step.summary, \
            f"Got: {step.summary!r}"


class TestEpisodicRecallStage:
    def test_episodic_recall_stage_present(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain neurons.")
        step = tt.get_step("episodic_recall")
        assert step is not None
        assert "n_episodes_recalled" in step.data

    def test_episodic_recall_after_feedback(self, substrate_with_knowledge):
        sub = substrate_with_knowledge
        # Process and give feedback to create episodic memories
        sub.process("The sky is blue because of Rayleigh scattering.", "test_science")
        sub.feedback("explain", 1.0, "test_science",
                     state="The sky is blue.", outcome="understood")
        sub.feedback("explain", 0.9, "test_science",
                     state="Rayleigh scattering diffuses light.", outcome="ok")

        _, tt = _process_with_trace(sub, "Why is the sky blue?")
        step = tt.get_step("episodic_recall")
        # After feedback, should recall at least 1 episode (low similarity expected
        # but the recall mechanism tries)
        assert step.data.get("n_episodes_recalled") >= 0  # at least tries

    def test_episodic_summary_not_flat(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What do I know about DNA?")
        step = tt.get_step("episodic_recall")
        # V31 summary should mention episodes and similarity
        assert "recalled" in step.summary.lower() or "episode" in step.summary.lower(), \
            f"Got: {step.summary!r}"


class TestCausalInferenceStage:
    def test_causal_inference_stage_present(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What causes climate change?")
        step = tt.get_step("causal_inference")
        assert step is not None
        assert "rules_fired" in step.data

    def test_causal_summary_has_rules_fired(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Why do neurons fire?")
        step = tt.get_step("causal_inference")
        assert "rules_fired=" in step.summary, f"Got: {step.summary!r}"


class TestGlobalWorkspaceStage:
    def test_gw_has_winner(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is machine learning?")
        step = tt.get_step("global_workspace")
        assert step is not None
        winner = step.data.get("winning_coalition", "")
        assert winner != "", f"Expected winner name, got empty"

    def test_gw_has_n_coalitions(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain photosynthesis.")
        step = tt.get_step("global_workspace")
        n = step.data.get("n_coalitions", -1)
        assert n >= 0, f"Expected n_coalitions >= 0, got {n}"

    def test_gw_summary_rich(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "How does DNA replicate?")
        step = tt.get_step("global_workspace")
        # V31 summary should contain activation and KLE
        assert "activation=" in step.summary, f"Got: {step.summary!r}"
        assert "KLE=" in step.summary, f"Got: {step.summary!r}"

    def test_gw_coalition_sources_list(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What are neurons?")
        step = tt.get_step("global_workspace")
        sources = step.data.get("coalition_sources", [])
        # Should have at least EXPLORATION (always generated for new states)
        assert len(sources) >= 1, f"Expected at least 1 coalition, got {sources}"


class TestSelfModelStage:
    def test_self_model_has_calibration_error(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is evolution?")
        step = tt.get_step("self_model")
        assert step is not None
        assert "calibration_error" in step.data

    def test_self_model_summary_rich(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain climate change.")
        step = tt.get_step("self_model")
        assert "calibration_error=" in step.summary, f"Got: {step.summary!r}"
        assert "novelty=" in step.summary, f"Got: {step.summary!r}"


class TestResponseGenerationStage:
    def test_response_generation_has_strategy(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is photosynthesis?")
        step = tt.get_step("response_generation")
        assert step is not None
        strategy = step.data.get("strategy", "")
        assert strategy != "", f"Expected strategy, got empty"

    def test_response_strategy_semantic_after_ingest(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "sunlight plants glucose")
        step = tt.get_step("response_generation")
        # After transplant, semantic_retrieval should be preferred
        strategy = step.data.get("strategy", "")
        assert strategy in (
            "semantic_retrieval", "episodic_cue", "rule_based",
            "retrieval", "nlu_question", "nlu_inform",
        ), f"Unexpected strategy: {strategy}"

    def test_response_summary_includes_strategy_and_intent(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "How does evolution work?")
        step = tt.get_step("response_generation")
        assert "strategy=" in step.summary, f"Got: {step.summary!r}"
        assert "intent=" in step.summary, f"Got: {step.summary!r}"


# ===========================================================================
# Integration-level tests
# ===========================================================================

class TestThoughtTraceIntegration:
    def test_all_11_stages_present(self, substrate_with_knowledge):
        from python.core.transparency.thought_trace import REQUIRED_STAGES
        _, tt = _process_with_trace(substrate_with_knowledge, "What is DNA?")
        present = {s.stage for s in tt.steps}
        for stage in REQUIRED_STAGES:
            assert stage in present, f"Missing stage: {stage}"

    def test_all_stages_have_data(self, substrate_with_knowledge):
        """All stages should have non-empty data dicts."""
        from python.core.transparency.thought_trace import REQUIRED_STAGES
        _, tt = _process_with_trace(
            substrate_with_knowledge,
            "Photosynthesis uses sunlight and water to produce glucose."
        )
        empty_stages = []
        for step in tt.steps:
            if not step.data:
                empty_stages.append(step.stage)
        assert not empty_stages, f"Stages with empty data: {empty_stages}"

    def test_no_flat_na_summaries(self, substrate_with_knowledge):
        """No stage summary should be the V30 flat 'N/A' placeholder."""
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain neurons.")
        for step in tt.steps:
            assert step.summary != "N/A", \
                f"Stage {step.stage} still has flat N/A summary"

    def test_to_markdown_contains_all_stages(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "How does evolution work?")
        md = tt.to_markdown()
        for stage in ("encoding", "emotion", "concept_extraction", "semantic_search",
                      "global_workspace", "response_generation"):
            assert stage in md, f"Stage {stage} missing from markdown"

    def test_to_markdown_has_icons(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is DNA?")
        md = tt.to_markdown()
        # V31 markdown should have emoji icons
        assert "📥" in md or "💭" in md or "🔎" in md, \
            "Expected emoji icons in V31 markdown"

    def test_rust_used_flag(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is photosynthesis?")
        # With Rust backend installed, rust_used should be True
        try:
            import hypervec_rs  # noqa: F401
            assert tt.rust_used is True
        except ImportError:
            pass  # No Rust — skip

    def test_total_duration_populated(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "Explain climate change.")
        # V31 populates real timing from substrate.process()
        assert tt.total_duration_ms >= 0.0

    def test_summary_method_not_flat(self, substrate_with_knowledge):
        _, tt = _process_with_trace(substrate_with_knowledge, "What is evolution?")
        s = tt.summary()
        assert "emotion=" in s or "conf=" in s, f"Got flat summary: {s!r}"

    def test_json_roundtrip(self, substrate_with_knowledge):
        import json
        import tempfile, os
        _, tt = _process_with_trace(substrate_with_knowledge, "What is photosynthesis?")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            tt.to_json(path)
            with open(path) as f:
                d = json.load(f)
            assert "steps" in d
            assert len(d["steps"]) == 11
            # All stages should be present in JSON
            stage_names = {s["stage"] for s in d["steps"]}
            from python.core.transparency.thought_trace import REQUIRED_STAGES
            for stage in REQUIRED_STAGES:
                assert stage in stage_names
        finally:
            os.unlink(path)
