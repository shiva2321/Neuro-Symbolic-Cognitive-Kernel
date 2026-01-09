"""
Neuromorphic Cognitive Graph Network (NCGN) - Complete Demonstration
Shows Phase 1-3 integration: Linguistic Graph + Spiking Neurons + Dual System
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from pathlib import Path
import yaml
import time

# NCGN Components
from ncgn.linguistic_graph import LinguisticGraph, GraphBuilder
from ncgn.graph_embeddings import GraphEmbedding, StructuralEncoder
from ncgn.spiking_neurons import LIFNeuron, PoissonEncoder, SpikingLayer
from ncgn.stdp_learning import STDPLearning, HomeostaticSTDP
from ncgn.graph_transformer import GraphTransformer
from ncgn.dual_system import DualSystemArchitecture

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/ncgn_config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def phase1_demo(config: dict):
    """
    Demonstrate Phase 1: Linguistic Graph Substrate
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: LINGUISTIC GRAPH SUBSTRATE")
    logger.info("=" * 80)

    # Sample corpus for demonstration
    corpus = [
        "The neural network learns patterns from data",
        "Deep learning uses neural networks with multiple layers",
        "Graph neural networks process graph-structured data",
        "Attention mechanisms help neural networks focus on relevant information",
        "Transformers use self-attention for sequence processing",
        "Spiking neural networks mimic biological neurons",
        "Cognitive architectures combine multiple reasoning systems",
        "Knowledge graphs represent structured information",
        "Machine learning algorithms learn from experience",
        "Artificial intelligence aims to create intelligent machines"
    ]

    # Build linguistic graph
    logger.info("\n--- Building Linguistic Graph ---")
    builder = GraphBuilder(
        embedding_model=config['linguistic_graph']['embedding_model'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    linguistic_graph = builder.build_from_corpus(
        corpus=corpus,
        window_size=config['linguistic_graph']['window_size'],
        min_frequency=1,
        max_vocab_size=config['linguistic_graph']['max_vocab_size']
    )

    # Display statistics
    stats = linguistic_graph.get_statistics()
    logger.info("\n--- Graph Statistics ---")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Save graph
    save_path = Path("saved_models/ncgn/linguistic_graph")
    linguistic_graph.save(save_path)
    logger.info(f"\n✓ Linguistic graph saved to {save_path}")

    return linguistic_graph


def phase2_demo(linguistic_graph: LinguisticGraph, config: dict):
    """
    Demonstrate Phase 2: Spiking Neural Network Integration
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: SPIKING NEURAL NETWORK INTEGRATION")
    logger.info("=" * 80)

    # Get node features from linguistic graph
    num_nodes = len(linguistic_graph.vocab)
    node_features = linguistic_graph.graph.ndata['feat'][:10]  # Use first 10 nodes for demo
    batch_size = 1

    logger.info(f"\n--- Setting up Spiking Network ---")
    logger.info(f"  Number of nodes: {node_features.shape[0]}")
    logger.info(f"  Feature dimension: {node_features.shape[1]}")

    # Create spiking layer
    in_features = node_features.shape[1]
    out_features = 128

    spiking_layer = SpikingLayer(
        in_features=in_features,
        out_features=out_features,
        neuron_model=config['spiking_network']['neuron_model'],
        bias=True
    )

    # Initialize STDP learning
    stdp = STDPLearning(
        tau_plus=config['stdp']['tau_plus'],
        tau_minus=config['stdp']['tau_minus'],
        A_plus=config['stdp']['A_plus'],
        A_minus=config['stdp']['A_minus']
    )

    # Encode features to spikes
    time_steps = config['spiking_network']['time_steps']
    logger.info(f"\n--- Encoding to Spike Trains ({time_steps} timesteps) ---")

    spike_trains = PoissonEncoder.encode(
        node_features.unsqueeze(0),
        time_steps=time_steps,
        max_rate=config['spiking_network']['max_firing_rate']
    )

    spike_count = spike_trains.sum().item()
    logger.info(f"  Total spikes generated: {spike_count}")
    logger.info(f"  Average firing rate: {spike_count / (time_steps * node_features.numel()):.4f}")

    # Simulate spiking network
    logger.info("\n--- Simulating Spiking Network ---")
    spiking_layer.reset_state(batch_size=batch_size, device='cpu')

    output_spikes_list = []

    start_time = time.time()
    for t in range(time_steps):
        input_spikes = spike_trains[t]
        output_spikes = spiking_layer(input_spikes)
        output_spikes_list.append(output_spikes)

        if t % 10 == 0:
            spike_rate = output_spikes.sum().item() / output_spikes.numel()
            logger.debug(f"  Step {t}: output spike rate = {spike_rate:.4f}")

    simulation_time = time.time() - start_time

    # Stack output spikes
    output_spike_train = torch.stack(output_spikes_list, dim=0)

    logger.info(f"  Simulation time: {simulation_time:.4f} seconds")
    logger.info(f"  Output spikes: {output_spike_train.sum().item()}")

    # Demonstrate STDP learning
    logger.info("\n--- STDP Learning ---")
    initial_weights = spiking_layer.weight.data.clone()

    for t in range(min(20, time_steps - 1)):
        pre_spikes = spike_trains[t]
        post_spikes = output_spike_train[t]

        dw = stdp.compute_weight_update(spiking_layer.weight.data, pre_spikes, post_spikes)
        spiking_layer.weight.data += dw

    weight_change = (spiking_layer.weight.data - initial_weights).abs().mean().item()
    logger.info(f"  Average weight change: {weight_change:.6f}")
    logger.info(f"✓ STDP learning demonstrated")

    return output_spike_train


def phase3_demo(linguistic_graph: LinguisticGraph, config: dict):
    """
    Demonstrate Phase 3: Dual-System Cognitive Architecture
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: DUAL-SYSTEM COGNITIVE ARCHITECTURE")
    logger.info("=" * 80)

    # Setup
    batch_size = 2
    num_nodes = min(50, len(linguistic_graph.vocab))
    node_features = linguistic_graph.graph.ndata['feat'][:num_nodes]

    # Create adjacency matrix
    edge_index = linguistic_graph.graph.edges()
    adjacency = torch.zeros(num_nodes, num_nodes)

    for i, (src, dst) in enumerate(zip(edge_index[0][:1000], edge_index[1][:1000])):
        if src < num_nodes and dst < num_nodes:
            adjacency[src, dst] = 1.0

    adjacency = adjacency.unsqueeze(0).repeat(batch_size, 1, 1)
    node_features = node_features.unsqueeze(0).repeat(batch_size, 1, 1)

    logger.info(f"\n--- Initializing Dual-System Architecture ---")
    logger.info(f"  Node features: {node_features.shape}")
    logger.info(f"  Adjacency: {adjacency.shape}")

    # Create dual-system architecture
    dual_system = DualSystemArchitecture(
        node_feat_dim=node_features.shape[-1],
        embed_dim=config['graph_transformer']['embed_dim'],
        num_layers=config['graph_transformer']['num_layers'],
        num_heads=config['graph_transformer']['num_heads'],
        reasoning_depth=config['symbolic_reasoner']['reasoning_depth'],
        integration_method=config['dual_system']['integration_method'],
        confidence_threshold=config['dual_system']['confidence_threshold']
    )

    # Add knowledge to System 2
    logger.info("\n--- Adding Knowledge to System 2 ---")
    knowledge_facts = [
        ("neural_network", "is_a", "machine_learning_model"),
        ("machine_learning_model", "is_a", "algorithm"),
        ("graph", "is_a", "data_structure"),
        ("transformer", "uses", "attention_mechanism"),
        ("spiking_neuron", "mimics", "biological_neuron")
    ]

    dual_system.add_knowledge(knowledge_facts)
    logger.info(f"  Added {len(knowledge_facts)} facts to knowledge base")

    # Test with different scenarios
    logger.info("\n--- Scenario 1: High Confidence (System 1 Only) ---")
    dual_system.confidence_threshold = 0.3

    output1 = dual_system(node_features, adjacency)

    logger.info(f"  Mode: {output1['mode']}")
    logger.info(f"  Confidence: {output1['confidence'].mean().item():.4f}")
    logger.info(f"  Output shape: {output1['output'].shape}")

    logger.info("\n--- Scenario 2: Low Confidence (Dual System) ---")
    dual_system.confidence_threshold = 0.8

    context = {'entity': 'neural_network'}
    output2 = dual_system(node_features, adjacency, context=context)

    logger.info(f"  Mode: {output2['mode']}")
    logger.info(f"  Confidence: {output2['confidence'].mean().item():.4f}")

    # Get explanation
    explanation = dual_system.explain_decision(output2)
    logger.info(f"\n  Explanation:")
    for line in explanation.split('\n'):
        logger.info(f"    {line}")

    logger.info(f"\n✓ Dual-system architecture demonstrated")

    return dual_system


def hardware_info():
    """Display hardware information"""
    logger.info("\n" + "=" * 80)
    logger.info("HARDWARE INFORMATION")
    logger.info("=" * 80)

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available")
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"  Current allocation: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
    else:
        logger.info("✗ CUDA not available (using CPU)")

    # Memory info
    import psutil
    memory = psutil.virtual_memory()
    logger.info(f"\n  System RAM: {memory.total / 1e9:.2f} GB")
    logger.info(f"  Available: {memory.available / 1e9:.2f} GB")
    logger.info(f"  Used: {memory.used / 1e9:.2f} GB ({memory.percent}%)")

    # CPU info
    logger.info(f"\n  CPU cores: {psutil.cpu_count()}")
    logger.info(f"  CPU usage: {psutil.cpu_percent()}%")


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='NCGN Demonstration')
    parser.add_argument('--config', type=str, default='configs/ncgn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                       help='Run specific phase only (default: all phases)')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1 and load existing graph')

    args = parser.parse_args()

    # Display banner
    print("\n" + "=" * 80)
    print("  NEUROMORPHIC COGNITIVE GRAPH NETWORK (NCGN)")
    print("  Multi-Phase Demonstration")
    print("=" * 80 + "\n")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"✓ Configuration loaded from {args.config}")

    # Display hardware info
    hardware_info()

    # Run phases
    linguistic_graph = None

    if args.phase is None or args.phase == 1:
        if args.skip_phase1:
            logger.info("\n⊳ Skipping Phase 1 - Loading existing graph...")
            try:
                linguistic_graph = LinguisticGraph.load("saved_models/ncgn/linguistic_graph")
                logger.info("✓ Graph loaded successfully")
            except:
                logger.warning("✗ Failed to load graph, will create new one")
                linguistic_graph = phase1_demo(config)
        else:
            linguistic_graph = phase1_demo(config)

    if args.phase is None or args.phase == 2:
        if linguistic_graph is None:
            logger.info("\n⊳ Loading linguistic graph for Phase 2...")
            linguistic_graph = LinguisticGraph.load("saved_models/ncgn/linguistic_graph")

        phase2_demo(linguistic_graph, config)

    if args.phase is None or args.phase == 3:
        if linguistic_graph is None:
            logger.info("\n⊳ Loading linguistic graph for Phase 3...")
            linguistic_graph = LinguisticGraph.load("saved_models/ncgn/linguistic_graph")

        dual_system = phase3_demo(linguistic_graph, config)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\n✓ Phase 1: Linguistic Graph Substrate - IMPLEMENTED")
    logger.info("✓ Phase 2: Spiking Neural Networks - IMPLEMENTED")
    logger.info("✓ Phase 3: Dual-System Architecture - IMPLEMENTED")
    logger.info("⊳ Phase 4: Continual Learning - PLANNED")
    logger.info("⊳ Phase 5: Hardware Optimization - PLANNED")

    logger.info("\n📊 Next Steps:")
    logger.info("  1. Implement Phase 4: Continual Graph Learning (CGL)")
    logger.info("  2. Implement Phase 5: Memory Optimization")
    logger.info("  3. Train on larger datasets")
    logger.info("  4. Benchmark on CGLB")
    logger.info("  5. Deploy optimization for RTX 3060 12GB")

    logger.info("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

