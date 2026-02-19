#!/usr/bin/env python3
"""
NSCK Autonomous Training Pipeline — Entry Point
=================================================

Launch the full autonomous training pipeline with a single command.

Usage::

    # Full training (fetches HuggingFace datasets, trains, evaluates)
    python nsck_ai_model/train_autonomous.py

    # Quick training (smaller dataset, faster)
    python nsck_ai_model/train_autonomous.py --quick

    # Text-only training (no images)
    python nsck_ai_model/train_autonomous.py --text-only

    # Interactive test after training
    python nsck_ai_model/train_autonomous.py --interactive

    # Custom limits
    python nsck_ai_model/train_autonomous.py --text-samples 5000 --image-samples 300
"""

import argparse
import json
import logging
import os
import sys
import time

# Ensure the workspace root is on sys.path
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from nsck_ai_model.autonomous_trainer import AutonomousTrainer, TrainingConfig
from nsck_ai_model.math_handler import is_math_question


def main():
    parser = argparse.ArgumentParser(
        description="NSCK Autonomous Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nsck_ai_model.train_autonomous               # Full training
  python -m nsck_ai_model.train_autonomous --quick        # Quick (500 text samples)
  python -m nsck_ai_model.train_autonomous --text-only    # No image training
  python -m nsck_ai_model.train_autonomous --interactive  # Interactive test after
  python -m nsck_ai_model.train_autonomous --eval-only    # Eval only (no training)
        """)

    # Training mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick training with reduced data")
    parser.add_argument("--text-only", action="store_true",
                        help="Train on text only (skip images)")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive test after training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (no training)")

    # Data limits
    parser.add_argument("--text-samples", type=int, default=3000,
                        help="Max text samples (default: 3000)")
    parser.add_argument("--qa-samples", type=int, default=500,
                        help="Max QA samples (default: 500)")
    parser.add_argument("--math-samples", type=int, default=300,
                        help="Max math samples (default: 300)")
    parser.add_argument("--image-samples", type=int, default=200,
                        help="Max image samples (default: 200)")
    parser.add_argument("--conversation-samples", type=int, default=300,
                        help="Max conversation samples (default: 300)")

    # Options
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation between phases")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Don't save checkpoints")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to JSON file")

    args = parser.parse_args()

    # Configure
    config = TrainingConfig(
        text_samples=args.text_samples,
        qa_samples=args.qa_samples,
        math_samples=args.math_samples,
        image_samples=args.image_samples,
        conversation_samples=args.conversation_samples,
        eval_after_phases=not args.no_eval,
        save_checkpoints=not args.no_checkpoint,
        log_level=args.log_level,
    )

    if args.text_only:
        config.phases = [
            "rich_corpus", "text_knowledge",
            "qa_training", "math_training",
            "conversation", "reinforcement",
        ]

    trainer = AutonomousTrainer(config=config)

    if args.eval_only:
        # Just evaluate
        print("\nRunning evaluation on untrained model...")
        report = trainer.evaluator.run_full_evaluation()
        print(report.summary())

        # Also check response quality
        quality = trainer.evaluator.evaluate_response_quality()
        print("\nResponse Quality:")
        for k, v in quality.items():
            print(f"  {k}: {v}")
        return

    # Run training
    if args.quick:
        report = trainer.quick_train(
            text_limit=min(args.text_samples, 500),
            image_limit=min(args.image_samples, 50))
    else:
        report = trainer.train()

    # Print report
    print(report.summary())

    # Save report if requested
    if args.output:
        report_data = {
            "total_duration_s": report.total_duration_s,
            "phases": [
                {
                    "phase": pr.phase,
                    "samples_trained": pr.samples_trained,
                    "duration_s": pr.duration_s,
                    "errors": pr.errors,
                    "eval_accuracy": (pr.eval_report.overall_accuracy
                                      if pr.eval_report else None),
                }
                for pr in report.phases
            ],
            "final_eval": {
                "accuracy": report.final_eval.overall_accuracy,
                "avg_score": report.final_eval.overall_avg_score,
                "total_questions": report.final_eval.total_questions,
                "categories": {
                    cat: {
                        "accuracy": score.accuracy,
                        "correct": score.correct,
                        "total": score.total,
                    }
                    for cat, score in report.final_eval.categories.items()
                },
            } if report.final_eval else None,
            "engine_stats": report.engine_stats,
        }
        with open(args.output, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")

    # Quick demo
    print("\n" + "="*60)
    print("  Quick Demo — Testing trained model")
    print("="*60 + "\n")

    demo_questions = [
        "What is the capital of France?",
        "What is 25 * 4?",
        "What is the largest ocean on Earth?",
        "What is photosynthesis?",
        "What is 15% of 200?",
        "Hello, how are you?",
        "What is the water cycle?",
        "Who developed the theory of relativity?",
    ]

    for q in demo_questions:
        try:
            # Math questions go through math handler
            if is_math_question(q):
                result = trainer.engine.chat(q, auto_learn=False)
            else:
                result = trainer.engine.chat(q, auto_learn=False)
            print(f"Q: {q}")
            print(f"A: {result['response']}")
            print(f"   [confidence={result['confidence']:.2f}]")
            print()
        except Exception as e:
            print(f"Q: {q}")
            print(f"A: ERROR — {e}")
            print()

    # Interactive mode
    if args.interactive:
        trainer.interactive_test()


if __name__ == "__main__":
    main()
