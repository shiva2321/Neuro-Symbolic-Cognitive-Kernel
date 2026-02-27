#!/usr/bin/env python3
"""Seed NSCK from a ConceptNet CSV file.

Usage:
    python nsck/scripts/seed_conceptnet.py <csv_path> [--output <out.kp>]
                                           [--max-concepts 50000]
                                           [--min-weight 2.0]
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Seed NSCK from ConceptNet CSV")
    parser.add_argument("csv_path", help="Path to ConceptNet assertions CSV (TSV format)")
    parser.add_argument("--output", default="nsck/data/knowledge_packs/conceptnet_en_50k.kp",
                        help="Output .kp file path")
    parser.add_argument("--max-concepts", type=int, default=50_000)
    parser.add_argument("--min-weight", type=float, default=2.0)
    args = parser.parse_args()

    from python.core.seeding.conceptnet_loader import ConceptNetLoader
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(
        args.csv_path,
        max_concepts=args.max_concepts,
        min_weight=args.min_weight,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pack.save(args.output)
    print(f"Saved {len(pack._concepts)} concepts to {args.output}")


if __name__ == "__main__":
    main()
