#!/usr/bin/env python3
"""Seed NSCK with BERT embeddings via the transplant pipeline.

Usage:
    python nsck/scripts/seed_bert.py [--model bert-base-uncased]
                                     [--output <out.kp>]
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Seed NSCK with BERT embeddings")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--output", default="nsck/data/knowledge_packs/bert_base_uncased.kp")
    args = parser.parse_args()

    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate
    from python.core.seeding.bert_seeder import BertSeeder

    cfg = NSCKConfig()
    substrate = NSCKSubstrate(config=cfg)
    seeder = BertSeeder()
    report = seeder.seed(substrate, model_name=args.model, save_pack=args.output)
    print(f"BERT seeding complete: {report}")


if __name__ == "__main__":
    main()
