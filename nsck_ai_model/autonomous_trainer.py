"""
NSCK Autonomous Training Pipeline
===================================

Multi-phase training orchestrator that automatically:
1. Fetches datasets from HuggingFace and built-in corpora
2. Trains the NSCK model on text, images, QA, math, and conversations
3. Evaluates after each phase using real benchmarks
4. Adapts training based on evaluation results
5. Saves checkpoints and produces a trained model

No neural networks — uses only VSA, symbolic processing,
and classical signal processing.

Usage::

    from nsck_ai_model.autonomous_trainer import AutonomousTrainer

    trainer = AutonomousTrainer()
    report = trainer.train()
    print(report)
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.data_fetcher import (
    DataFetcher, TextSample, QASample, MathSample,
    ImageTextSample, ConversationSample,
    stream_rich_corpus,
    fetch_wikitext, fetch_dolly_conversations, fetch_gsm8k,
    fetch_openbookqa, fetch_arc,
)
from nsck_ai_model.math_handler import MathHandler, is_math_question
from nsck_ai_model.evaluator import ModelEvaluator, EvalReport

logger = logging.getLogger("nsck_ai.trainer")


# ═══════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Configuration for the autonomous training pipeline."""

    # Data fetching limits
    text_samples: int = 3000
    qa_samples: int = 500
    math_samples: int = 300
    image_samples: int = 200
    conversation_samples: int = 300

    # Training phases
    phases: List[str] = field(default_factory=lambda: [
        "rich_corpus",          # Built-in rich knowledge
        "text_knowledge",       # WikiText + TinyStories
        "qa_training",          # OpenBookQA + ARC
        "math_training",        # GSM8K word problems
        "conversation",         # Dolly-15k conversations
        "image_training",       # CIFAR-10 image-caption pairs
        "reinforcement",        # Re-train on weak areas
    ])

    # Evaluation
    eval_after_phases: bool = True
    eval_categories: Optional[List[str]] = None

    # Checkpointing
    checkpoint_dir: str = ""
    save_checkpoints: bool = True

    # Logging
    log_level: str = "INFO"
    progress_interval: int = 100     # Report every N samples

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(
                os.path.dirname(__file__), ".checkpoints")


# ═══════════════════════════════════════════════════════════════════════════
# Phase Results
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    """Result of one training phase."""
    phase: str
    samples_trained: int
    duration_s: float
    eval_report: Optional[EvalReport] = None
    errors: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingReport:
    """Complete training session report."""
    start_time: float
    end_time: float
    total_duration_s: float
    phases: List[PhaseResult]
    final_eval: Optional[EvalReport] = None
    engine_stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  NSCK AUTONOMOUS TRAINING REPORT",
            f"{'='*70}",
            f"  Total duration:  {self.total_duration_s:.1f}s "
            f"({self.total_duration_s/60:.1f} min)",
            f"  Training phases: {len(self.phases)}",
            f"",
        ]

        total_samples = 0
        for pr in self.phases:
            total_samples += pr.samples_trained
            acc = ""
            if pr.eval_report:
                acc = f"  eval={pr.eval_report.overall_accuracy:.1%}"
            lines.append(
                f"  {pr.phase:25s}  "
                f"samples={pr.samples_trained:5d}  "
                f"time={pr.duration_s:6.1f}s  "
                f"errors={pr.errors}"
                f"{acc}")

        lines.append(f"")
        lines.append(f"  Total samples trained: {total_samples}")

        if self.final_eval:
            lines.append(f"")
            lines.append(f"  FINAL EVALUATION:")
            lines.append(f"    Accuracy:    {self.final_eval.overall_accuracy:.1%}")
            lines.append(f"    Avg score:   {self.final_eval.overall_avg_score:.3f}")
            lines.append(f"    Questions:   {self.final_eval.total_questions}")
            lines.append(f"")
            for cat, score in sorted(self.final_eval.categories.items()):
                lines.append(
                    f"    {cat:25s}  "
                    f"{score.accuracy:6.1%}  "
                    f"({score.correct}/{score.total})")

        if self.engine_stats:
            lines.append(f"")
            lines.append(f"  ENGINE STATE:")
            knowledge = self.engine_stats.get('knowledge', {})
            gen = self.engine_stats.get('generator', {})
            lines.append(f"    Concepts:     {knowledge.get('total_concepts', 0)}")
            lines.append(f"    Relations:    {knowledge.get('total_relations', 0)}")
            lines.append(f"    Facts:        {knowledge.get('total_facts', 0)}")
            lines.append(f"    Sentences:    {knowledge.get('sentence_store', 0)}")
            lines.append(f"    Bigram vocab: {gen.get('bigram_vocab', 0)}")
            lines.append(f"    Corpus size:  {gen.get('corpus_size', 0)}")

        lines.append(f"{'='*70}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main Autonomous Trainer
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousTrainer:
    """Autonomous training pipeline for the NSCK AI model.

    Fetches datasets, trains the model phase by phase, evaluates
    performance, and adapts training to improve weak areas.

    Architecture:
        - NO neural networks
        - Uses VSA (10,240-bit binary hypervectors) for all representations
        - Uses classical signal processing for images
        - Uses symbolic computation for math
        - All knowledge is stored in semantic memory + episodic memory
        - Response generation uses n-gram models + knowledge retrieval

    Usage::

        trainer = AutonomousTrainer()
        report = trainer.train()
        print(report.summary())

        # The trained engine is accessible:
        engine = trainer.engine
        result = engine.chat("What is the capital of France?")
    """

    def __init__(
        self,
        engine: Optional[NSCKAIEngine] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.engine = engine or NSCKAIEngine()
        self.fetcher = DataFetcher()
        self.math_handler = MathHandler()
        self.evaluator = ModelEvaluator(self.engine)
        self._phase_results: List[PhaseResult] = []
        self._start_time = 0.0

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        # Set up checkpoint directory
        if self.config.save_checkpoints:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    # ─── Main entry point ─────────────────────────────────────────────

    def train(self) -> TrainingReport:
        """Run the full autonomous training pipeline.

        Returns:
            TrainingReport with all phase results and final evaluation.
        """
        self._start_time = time.time()
        logger.info("="*60)
        logger.info("  NSCK Autonomous Training Pipeline — Starting")
        logger.info("="*60)

        for phase_idx, phase in enumerate(self.config.phases):
            phase_start = time.time()
            logger.info("─" * 50)
            logger.info("Phase [%d/%d]: %s",
                        phase_idx + 1, len(self.config.phases), phase)
            try:
                result = self._run_phase(phase)
                self._phase_results.append(result)
                logger.info(
                    "Phase [%s] complete: %d samples, %.1fs, %d errors",
                    phase, result.samples_trained,
                    result.duration_s, result.errors)

                if self.config.eval_after_phases:
                    eval_report = self.evaluator.run_full_evaluation()
                    result.eval_report = eval_report
                    logger.info(
                        "Post-phase eval: accuracy=%.1f%% score=%.3f",
                        eval_report.overall_accuracy * 100,
                        eval_report.overall_avg_score)

            except Exception as e:
                logger.error("Phase [%s] failed: %s", phase, e)
                logger.error(traceback.format_exc())
                self._phase_results.append(PhaseResult(
                    phase=phase, samples_trained=0,
                    duration_s=0.0, errors=1,
                    details={"error": str(e)}))

        # Finalize learned models (build VSA prototypes from training data)
        logger.info("Finalizing learned cognitive models...")
        self.engine.finalize_training()

        # Final comprehensive evaluation
        logger.info("Running final evaluation...")
        final_eval = self.evaluator.run_full_evaluation()

        # Save checkpoint
        if self.config.save_checkpoints:
            self._save_checkpoint("final")

        end_time = time.time()
        total_duration = end_time - self._start_time

        # Build report
        report = TrainingReport(
            start_time=self._start_time,
            end_time=end_time,
            total_duration_s=total_duration,
            phases=self._phase_results,
            final_eval=final_eval,
            engine_stats=self.engine.get_system_stats(),
        )

        logger.info(report.summary())
        return report

    # ─── Phase dispatcher ─────────────────────────────────────────────

    def _run_phase(self, phase: str) -> PhaseResult:
        """Run a single training phase."""
        dispatch = {
            "rich_corpus": self._phase_rich_corpus,
            "text_knowledge": self._phase_text_knowledge,
            "qa_training": self._phase_qa_training,
            "math_training": self._phase_math_training,
            "conversation": self._phase_conversation,
            "image_training": self._phase_image_training,
            "reinforcement": self._phase_reinforcement,
        }

        handler = dispatch.get(phase)
        if handler is None:
            logger.warning("Unknown phase: %s", phase)
            return PhaseResult(phase=phase, samples_trained=0,
                               duration_s=0.0, errors=1)
        return handler()

    # ─── Phase: Rich Corpus (built-in knowledge) ──────────────────────

    def _phase_rich_corpus(self) -> PhaseResult:
        """Train on the built-in rich knowledge corpus."""
        logger.info("Phase: Rich Corpus (built-in knowledge)")
        start = time.time()
        count = 0
        errors = 0

        for sample in stream_rich_corpus():
            try:
                self.engine.train_on_text(sample.text, source_quality=2.0)
                count += 1
            except Exception as e:
                errors += 1
                logger.debug("Rich corpus error: %s", e)

        return PhaseResult(
            phase="rich_corpus",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors,
            details={"source": "built_in"})

    # ─── Phase: Text Knowledge (HuggingFace) ──────────────────────────

    def _phase_text_knowledge(self) -> PhaseResult:
        """Train on text from HuggingFace datasets."""
        logger.info("Phase: Text Knowledge (HuggingFace)")
        start = time.time()
        count = 0
        errors = 0

        sources = ["wikitext", "tinystories", "dolly"]
        limit = self.config.text_samples

        for sample in self.fetcher.fetch_text(limit=limit, sources=sources):
            try:
                self.engine.train_on_text(sample.text, source_quality=0.5)
                count += 1
                if count % self.config.progress_interval == 0:
                    logger.info(
                        "  Text knowledge: %d/%d samples trained",
                        count, limit)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.debug("Text training error: %s", e)

        return PhaseResult(
            phase="text_knowledge",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors,
            details={"fetch_stats": self.fetcher.get_stats()})

    # ─── Phase: QA Training ───────────────────────────────────────────

    def _phase_qa_training(self) -> PhaseResult:
        """Train on Q&A pairs — teaches the model factual knowledge
        and learns Q&A response patterns for the ResponseComposer."""
        logger.info("Phase: QA Training")
        start = time.time()
        count = 0
        errors = 0

        for sample in self.fetcher.fetch_qa(limit=self.config.qa_samples):
            try:
                # Train the answer as knowledge text
                knowledge_text = f"{sample.question} {sample.answer}"
                self.engine.train_on_text(knowledge_text, source_quality=1.5)

                # Also train just the answer for retrieval
                if len(sample.answer) > 20:
                    self.engine.train_on_text(sample.answer, source_quality=1.5)

                # Teach the response composer Q&A patterns
                self.engine.response_composer.learn_from_qa(
                    sample.question, sample.answer)

                count += 1
                if count % self.config.progress_interval == 0:
                    logger.info("  QA: %d samples trained", count)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.debug("QA training error: %s", e)

        return PhaseResult(
            phase="qa_training",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors)

    # ─── Phase: Math Training ─────────────────────────────────────────

    def _phase_math_training(self) -> PhaseResult:
        """Train on math word problems.

        The NSCK model learns mathematical language, operator words, and
        relationships from GSM8K.  The ``MathHandler`` learns operator
        mappings from solved examples.  Actual computation remains symbolic.
        """
        logger.info("Phase: Math Training")
        start = time.time()
        count = 0
        errors = 0

        # Train on GSM8K word problems (learns operator words too)
        for sample in self.fetcher.fetch_math(
                limit=self.config.math_samples):
            try:
                # Learn the word problem and its solution steps
                full_text = f"{sample.question} The answer is {sample.answer}."
                if sample.solution_steps:
                    full_text = (f"{sample.question} "
                                 f"{sample.solution_steps} "
                                 f"The answer is {sample.answer}.")
                self.engine.train_on_text(full_text, source_quality=1.0)

                # Teach the math handler's operator detector from this example
                self.engine.math_handler.learn(full_text)
                try:
                    ans_val = float(
                        sample.answer.replace(',', '').strip().split()[0])
                    self.engine.math_handler.learn_from_word_problem(
                        sample.question, ans_val)
                except (ValueError, IndexError):
                    pass

                count += 1
                if count % self.config.progress_interval == 0:
                    logger.info("  Math: %d samples trained", count)
            except Exception as e:
                errors += 1

        return PhaseResult(
            phase="math_training",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors)

    # ─── Phase: Conversation Training ─────────────────────────────────

    def _phase_conversation(self) -> PhaseResult:
        """Train on conversational patterns from Dolly-15k.

        Also teaches the engine's intent classifier by providing direct
        conversational examples (greetings, thanks, questions), not just
        meta-descriptions about them.
        """
        logger.info("Phase: Conversation Training")
        start = time.time()
        count = 0
        errors = 0

        # Sub-phase: teach the intent classifier direct conversational
        # examples.  These are SHORT example inputs with their category.
        _GREETING_EXAMPLES = [
            "Hello!", "Hi there!", "Hey, how are you?",
            "Good morning!", "Good afternoon!", "Good evening!",
            "Hello, how are you?", "Hi!", "Hey!", "Howdy!",
            "Greetings!", "Hello, nice to meet you.",
        ]
        _THANKS_EXAMPLES = [
            "Thank you!", "Thanks!", "Thank you very much!",
            "Thanks for your help!", "Thank you for your help!",
            "I appreciate it!", "Thanks a lot!",
        ]
        _YESNO_EXAMPLES = [
            "Do whales breathe air?", "Is the sky blue?",
            "Does water boil at 100 degrees?", "Can fish swim?",
            "Are mammals warm-blooded?", "Is the Earth round?",
            "Do plants need sunlight?", "Does correlation imply causation?",
            "Have humans landed on the Moon?", "Will it rain tomorrow?",
            "Can computers think?", "Should we recycle?",
        ]
        _QUESTION_EXAMPLES = [
            "What is photosynthesis?", "How does the heart work?",
            "What is the capital of France?", "Why is the sky blue?",
            "What are cells?", "How do computers work?",
            "What is machine learning?", "What is evolution?",
        ]
        _GREETING_RESPONSES = [
            "I am fine, thank you for asking.",
            "I am doing well, thank you.",
            "Hello! I am fine, thank you.",
        ]
        _THANKS_RESPONSES = [
            "You are welcome.",
            "You are welcome, glad I could help.",
        ]

        for text in _GREETING_EXAMPLES:
            hv = self.engine._encode_text(text)
            self.engine.intent_memory.add_example(
                'greeting', hv, _GREETING_RESPONSES[0])
        for text in _THANKS_EXAMPLES:
            hv = self.engine._encode_text(text)
            self.engine.intent_memory.add_example(
                'thanks', hv, _THANKS_RESPONSES[0])
        for text in _YESNO_EXAMPLES:
            hv = self.engine._encode_text(text)
            self.engine.intent_memory.add_example('question_yesno', hv)
        for text in _QUESTION_EXAMPLES:
            hv = self.engine._encode_text(text)
            self.engine.intent_memory.add_example('question', hv)
        count += len(_GREETING_EXAMPLES) + len(_THANKS_EXAMPLES) + \
            len(_YESNO_EXAMPLES) + len(_QUESTION_EXAMPLES)

        # Sub-phase: train on Dolly conversations (learn Q&A patterns)
        for sample in self.fetcher.fetch_conversations(
                limit=self.config.conversation_samples):
            try:
                turns = sample.turns
                for i, turn in enumerate(turns):
                    content = turn.get('content', '')
                    if content and len(content) > 20:
                        self.engine.train_on_text(content, source_quality=0.8)
                        count += 1

                        # Teach intent from Q&A pairs
                        if (turn.get('role') == 'user'):
                            q_hv = self.engine._encode_text(content)
                            if content.strip().endswith('?'):
                                # Classify into yesno vs general question
                                first_word = content.strip().split()[0].lower() if content.strip().split() else ''
                                _AUX = {
                                    'do', 'does', 'did', 'is', 'are',
                                    'was', 'were', 'can', 'could',
                                    'would', 'should', 'has', 'have',
                                    'will', 'shall', 'may', 'might',
                                }
                                if first_word in _AUX:
                                    self.engine.intent_memory.add_example(
                                        'question_yesno', q_hv)
                                else:
                                    self.engine.intent_memory.add_example(
                                        'question', q_hv)

                        # Learn Q&A patterns from user→assistant pairs
                        if (turn.get('role') == 'assistant' and i > 0
                                and turns[i-1].get('role') == 'user'):
                            q = turns[i-1].get('content', '')
                            a = content
                            if q and a:
                                self.engine.response_composer.learn_from_qa(
                                    q, a)
            except Exception:
                errors += 1

        return PhaseResult(
            phase="conversation",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors)

    # ─── Phase: Image Training ────────────────────────────────────────

    def _phase_image_training(self) -> PhaseResult:
        """Train on image-caption pairs from CIFAR-10."""
        logger.info("Phase: Image Training")
        start = time.time()
        count = 0
        errors = 0

        for sample in self.fetcher.fetch_images(
                limit=self.config.image_samples):
            try:
                self.engine.train_on_image(sample.image, sample.caption)
                count += 1
                if count % 50 == 0:
                    logger.info("  Images: %d trained", count)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.debug("Image training error: %s", e)

        return PhaseResult(
            phase="image_training",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors)

    # ─── Phase: Reinforcement (re-train weak areas) ───────────────────

    def _phase_reinforcement(self) -> PhaseResult:
        """Re-train on areas where the model performed poorly.

        Strategy:
        1. Run evaluation on all categories.
        2. Identify weak categories (accuracy < 60%).
        3. Fetch additional targeted data for weak areas.
        4. Re-train.
        """
        logger.info("Phase: Reinforcement (adaptive re-training)")
        start = time.time()
        count = 0
        errors = 0

        # Evaluate current performance
        eval_report = self.evaluator.run_full_evaluation()
        weak_categories = []
        for cat, score in eval_report.categories.items():
            if score.accuracy < 0.6:
                weak_categories.append((cat, score.accuracy))
                logger.info("  Weak category: %s (%.1f%%)",
                            cat, score.accuracy * 100)

        if not weak_categories:
            logger.info("  No weak categories found — skipping reinforcement")
            return PhaseResult(
                phase="reinforcement",
                samples_trained=0,
                duration_s=time.time() - start,
                details={"skipped": True,
                         "all_categories_above_60pct": True})

        # Re-train on weak areas
        for cat, acc in weak_categories:
            try:
                additional = self._get_reinforcement_data(cat)
                for text in additional:
                    self.engine.train_on_text(text, source_quality=1.5)
                    count += 1
            except Exception as e:
                errors += 1
                logger.debug("Reinforcement error: %s", e)

        return PhaseResult(
            phase="reinforcement",
            samples_trained=count,
            duration_s=time.time() - start,
            errors=errors,
            details={
                "weak_categories": weak_categories,
                "pre_reinforcement_accuracy": eval_report.overall_accuracy,
            })

    def _get_reinforcement_data(self, category: str) -> List[str]:
        """Fetch additional training data for a weak category.

        Instead of hardcoded sentences, re-fetches *real* data from
        HuggingFace datasets targeted at the weak area.
        """
        data: List[str] = []
        limit = 50  # small batch of reinforcement data

        try:
            if category == "factual":
                # More encyclopedia text from WikiText
                for sample in fetch_wikitext(limit=limit):
                    data.append(sample.text)

            elif category == "reasoning":
                # ARC reasoning questions (answer as text)
                for sample in fetch_arc(limit=limit):
                    data.append(f"{sample.question} {sample.answer}")

            elif category == "math":
                # More GSM8K problems
                for sample in fetch_gsm8k(limit=limit):
                    text = f"{sample.question} The answer is {sample.answer}."
                    data.append(text)
                    # Also teach the math operator detector
                    self.engine.math_handler.learn(text)

            elif category == "conversation":
                # More Dolly conversation content
                for sample in fetch_dolly_conversations(limit=limit):
                    data.append(sample.text)

            elif category == "science":
                # OpenBookQA science facts
                for sample in fetch_openbookqa(limit=limit):
                    data.append(f"{sample.question} {sample.answer}")

            elif category == "technology":
                # Technology-related Dolly entries
                for sample in fetch_dolly_conversations(limit=limit):
                    text_lower = sample.text.lower()
                    if any(w in text_lower for w in [
                            'computer', 'software', 'program', 'internet',
                            'technology', 'digital', 'algorithm', 'data',
                            'network', 'system', 'code', 'machine']):
                        data.append(sample.text)
        except Exception as e:
            logger.warning("Reinforcement fetch for %s failed: %s",
                           category, e)

        # If online fetch yielded nothing, use the rich corpus as fallback
        if not data:
            for sample in stream_rich_corpus():
                if category.lower() in sample.category.lower():
                    data.append(sample.text)

        return data

    # ─── Checkpointing ────────────────────────────────────────────────

    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        try:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, f"checkpoint_{name}.json")

            knowledge = self.engine.export_knowledge()
            stats = self.engine.get_system_stats()

            checkpoint = {
                "name": name,
                "timestamp": time.time(),
                "stats": stats,
                "knowledge_summary": {
                    "concepts": len(knowledge.get('concepts', {})),
                    "relations": len(knowledge.get('relations', [])),
                    "facts": len(knowledge.get('facts', [])),
                },
                "phase_results": [
                    {
                        "phase": pr.phase,
                        "samples": pr.samples_trained,
                        "duration_s": pr.duration_s,
                        "errors": pr.errors,
                        "eval_accuracy": (pr.eval_report.overall_accuracy
                                          if pr.eval_report else None),
                    }
                    for pr in self._phase_results
                ],
            }

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)

            logger.info("Checkpoint saved: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    # ─── Convenience: quick train + test ──────────────────────────────

    def quick_train(self, text_limit: int = 200,
                    image_limit: int = 50) -> TrainingReport:
        """Run a quick training session with reduced data.

        Good for testing and iterative development.  Mid-phase
        evaluation is disabled to avoid slowdowns — a final
        evaluation is always run at the end.
        """
        self.config.text_samples = text_limit
        self.config.qa_samples = min(100, text_limit // 5)
        self.config.math_samples = min(50, text_limit // 10)
        self.config.image_samples = image_limit
        self.config.conversation_samples = min(100, text_limit // 5)
        # Disable mid-phase evaluation to keep quick_train fast
        self.config.eval_after_phases = False
        return self.train()

    def train_text_only(self, limit: int = 2000) -> TrainingReport:
        """Train on text only (no images)."""
        self.config.phases = [
            "rich_corpus", "text_knowledge",
            "qa_training", "math_training",
            "conversation", "reinforcement",
        ]
        self.config.text_samples = limit
        return self.train()

    def interactive_test(self):
        """Run an interactive test session after training.

        Press Ctrl+C to exit.
        """
        print("\n" + "="*60)
        print("  NSCK AI Interactive Test Session")
        print("  Type a question or statement. Type 'quit' to exit.")
        print("  Type 'stats' to see model statistics.")
        print("  Type 'eval' to run evaluation.")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                if user_input.lower() == 'stats':
                    stats = self.engine.get_system_stats()
                    print(json.dumps(stats, indent=2, default=str))
                    continue
                if user_input.lower() == 'eval':
                    report = self.evaluator.run_full_evaluation()
                    print(report.summary())
                    continue

                # Check if it's a math question
                if is_math_question(user_input):
                    math_result = self.math_handler.solve(user_input)
                    if math_result.confidence > 0.5:
                        print(f"AI:  {math_result.answer}")
                        if math_result.steps:
                            print(f"     Steps: {' → '.join(math_result.steps)}")
                        continue

                result = self.engine.chat(user_input)
                print(f"AI:  {result['response']}")
                print(f"     [confidence={result['confidence']:.2f}, "
                      f"emotion={result['emotion']['emotion']}, "
                      f"latency={result['latency_ms']:.0f}ms]")
            except KeyboardInterrupt:
                print("\nExiting interactive test.")
                break
            except Exception as e:
                print(f"Error: {e}")
