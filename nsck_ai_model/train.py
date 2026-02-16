#!/usr/bin/env python3
"""
NSCK AI Training Script
=======================

End-to-end training of the NSCK AI model on real data.

Usage::

    # Quick training on seed corpus only
    python -m nsck_ai_model.train --mode seed

    # Full training with HuggingFace data
    python -m nsck_ai_model.train --mode full --hf-limit 500

    # Train and start interactive chat
    python -m nsck_ai_model.train --mode seed --chat
"""

import argparse
import logging
import time
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.data_pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("nsck_ai.train")


def train(mode: str = "seed", hf_limit: int = 500) -> NSCKAIEngine:
    """Train a fresh engine and return it."""
    engine = NSCKAIEngine()
    pipeline = DataPipeline(engine)

    logger.info("=" * 60)
    logger.info("NSCK AI Model — Training")
    logger.info("=" * 60)

    if mode == "seed":
        logger.info("Mode: seed corpus only (fast, offline)")
        stats = pipeline.train_from_seed()
    elif mode == "hf":
        logger.info("Mode: HuggingFace data only (limit=%d)", hf_limit)
        stats = pipeline.train_from_hf(limit=hf_limit)
    elif mode == "full":
        logger.info("Mode: seed + HuggingFace (limit=%d)", hf_limit)
        stats = pipeline.train_general(hf_limit=hf_limit)
    else:
        logger.error("Unknown mode: %s", mode)
        sys.exit(1)

    logger.info("Training complete: %s", stats)

    # Print summary
    sys_stats = engine.get_system_stats()
    logger.info("-" * 40)
    logger.info("Concepts learned : %d",
                sys_stats['knowledge']['total_concepts'])
    logger.info("Relations learned: %d",
                sys_stats['knowledge']['total_relations'])
    logger.info("Episodes stored  : %d",
                sys_stats['knowledge']['total_episodes'])
    logger.info("Causal rules     : %d",
                sys_stats['causal_rules']['total_rules'])
    logger.info("Abstractions     : %d",
                sys_stats['abstractions']['total_abstractions'])
    logger.info("Vocabulary       : %d words",
                sys_stats['encoder']['vocabulary_size'])
    logger.info("Bigram vocab     : %d",
                sys_stats['generator']['bigram_vocab'])
    logger.info("Training time    : %.2fs",
                sys_stats['training']['training_time_s'])
    logger.info("-" * 40)

    return engine


def interactive_chat(engine: NSCKAIEngine):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("NSCK AI Chat — type 'quit' to exit, 'stats' for system info")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if user_input.lower() == 'stats':
            stats = engine.get_system_stats()
            for section, data in stats.items():
                print(f"\n  {section}:")
                if isinstance(data, dict):
                    for k, v in data.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"    {data}")
            continue
        if user_input.lower() == 'trace':
            print("  (Trace will be shown for the next query)")
            user_input = input("You: ").strip()
            if not user_input:
                continue
            result = engine.chat(user_input)
            print(f"\nAI: {result['response']}")
            print(f"\n--- Thought Trace ({result['trace']['step_count']} steps, "
                  f"{result['trace']['total_ms']:.1f}ms) ---")
            for step in result['trace']['steps']:
                print(f"  [{step['stage']}] {step['action']}"
                      f" ({step['duration_ms']:.1f}ms)")
                if step['outputs']:
                    for k, v in step['outputs'].items():
                        print(f"    → {k}: {v}")
            print()
            continue

        result = engine.chat(user_input)
        print(f"AI: {result['response']}")
        print(f"    (confidence={result['confidence']}, "
              f"emotion={result['emotion']['emotion']}, "
              f"latency={result['latency_ms']}ms)\n")


def main():
    parser = argparse.ArgumentParser(description="NSCK AI Model Training")
    parser.add_argument("--mode", choices=["seed", "hf", "full"],
                        default="seed",
                        help="Training mode: seed (offline), hf (HuggingFace), "
                             "full (both)")
    parser.add_argument("--hf-limit", type=int, default=500,
                        help="Max passages to stream from HuggingFace")
    parser.add_argument("--chat", action="store_true",
                        help="Start interactive chat after training")
    args = parser.parse_args()

    engine = train(mode=args.mode, hf_limit=args.hf_limit)

    if args.chat:
        interactive_chat(engine)


if __name__ == "__main__":
    main()
