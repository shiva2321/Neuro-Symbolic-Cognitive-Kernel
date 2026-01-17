import argparse
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.flash import FlashColony


def parse_args():
    parser = argparse.ArgumentParser(description="Million-neuron flash stress test")
    parser.add_argument("--filepath", default="stress_brain.dat", help="Binary brain file path")
    parser.add_argument("--neurons", type=int, default=1_000_000, help="Number of neurons to allocate")
    parser.add_argument("--synapses", type=int, default=5_000_000, help="Total synapses to wire")
    parser.add_argument("--max-edges-per-node", dest="max_edges_per_node", type=int, default=8,
                        help="Fixed synapse slots per neuron")
    parser.add_argument("--stimulate-pct", dest="stimulate_pct", type=float, default=0.01,
                        help="Fraction of neurons externally stimulated each tick")
    parser.add_argument("--ticks", type=int, default=10, help="Ticks to run during the benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing brain file instead of recreating")
    return parser.parse_args()


def ensure_brain_file(args) -> FlashColony:
    if os.path.exists(args.filepath) and not args.reuse:
        os.remove(args.filepath)
    colony = FlashColony(args.filepath, max_nodes=args.neurons, max_edges_per_node=args.max_edges_per_node)
    if not os.path.exists(args.filepath):
        colony.create_new(max_nodes=args.neurons)
    else:
        colony.connect_to_file()
    return colony


def allocate_nodes(colony: FlashColony, total_nodes: int):
    print(f"Allocating {total_nodes:,} neurons...")
    for nid in range(total_nodes):
        colony.add_node(nid, node_type="hidden", threshold=1.0)
        if (nid + 1) % 100_000 == 0:
            print(f"  added {nid + 1:,} neurons...")
    print("✓ Neuron allocation complete")


def plan_edge_counts(total_nodes: int, total_edges: int, max_per_node: int, rng: random.Random):
    if total_edges > total_nodes * max_per_node:
        raise ValueError("Requested synapses exceed per-node capacity; raise max-edges-per-node or reduce synapses.")
    counts = [0] * total_nodes
    for _ in range(total_edges):
        while True:
            nid = rng.randrange(total_nodes)
            if counts[nid] < max_per_node:
                counts[nid] += 1
                break
    return counts


def wire_edges(colony: FlashColony, counts, total_nodes: int, rng: random.Random):
    print("Wiring synapses...")
    wired = 0
    for src, count in enumerate(counts):
        if count == 0:
            continue
        for _ in range(count):
            tgt = rng.randrange(total_nodes)
            weight = rng.uniform(0.2, 1.0)
            colony.connect_binary(src, tgt, weight)
            wired += 1
        if wired and wired % 500_000 == 0:
            print(f"  wired {wired:,} synapses...")
    print(f"✓ Synapse wiring complete ({wired:,} total)")


def stimulate_and_benchmark(colony: FlashColony, args) -> tuple[float, float]:
    rng = random.Random(args.seed + 1)
    stim_per_tick = max(1, int(args.neurons * args.stimulate_pct))
    print(f"Stimulating {stim_per_tick:,} neurons per tick for {args.ticks} ticks...")
    t0 = time.perf_counter()
    for tick in range(args.ticks):
        for _ in range(stim_per_tick):
            colony.set_input(rng.randrange(args.neurons), 1.5)
        colony.step(global_dopamine=0.0, learning_rate=0.0)
        if (tick + 1) % 5 == 0:
            print(f"  completed tick {tick + 1}")
    duration = time.perf_counter() - t0
    tps = args.ticks / duration if duration > 0 else 0.0
    return tps, duration


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    colony = ensure_brain_file(args)
    try:
        allocate_nodes(colony, args.neurons)
        counts = plan_edge_counts(args.neurons, args.synapses, args.max_edges_per_node, rng)
        wire_edges(colony, counts, args.neurons, rng)

        file_size_mb = os.path.getsize(args.filepath) / (1024 * 1024)
        print(f"Brain file size: {file_size_mb:.2f} MB")

        tps, duration = stimulate_and_benchmark(colony, args)
        print("\n=== Stress Test Results ===")
        print(f"Neurons:                {args.neurons:,}")
        print(f"Synapses:               {args.synapses:,}")
        print(f"Max edges per neuron:   {args.max_edges_per_node}")
        print(f"Ticks run:              {args.ticks}")
        print(f"Stimulated per tick:    {max(1, int(args.neurons * args.stimulate_pct)):,}")
        print(f"Wall time:              {duration:.2f} s")
        print(f"Ticks per second (TPS): {tps:,.2f}")
    finally:
        colony.close()


if __name__ == "__main__":
    main()

