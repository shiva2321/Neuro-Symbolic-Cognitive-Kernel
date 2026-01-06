"""
AI-Enhanced Training Example
Demonstrates how to use Claude and Gemini to improve neural network training
"""

import os
from core.central_controller import CentralController
from training.ai_training_assistant import setup_ai_training, AIEnhancedTrainingEngine
from training.dataset_loader import DatasetLoader
from core.graph_network import GraphNetwork
from utils.text_processor import TextProcessor


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           AI-ENHANCED NEURAL NETWORK TRAINING                        ║
║                                                                      ║
║  Powered by Claude (Anthropic) and Gemini (Google)                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Setup API keys
    print("\n[Step 1] Setting up AI APIs...")
    print("\nYou need API keys from:")
    print("  • Claude (Anthropic): https://console.anthropic.com/")
    print("  • Gemini (Google): https://makersuite.google.com/app/apikey")
    print("\nOptions to provide API keys:")
    print("  1. Set environment variables:")
    print("     - ANTHROPIC_API_KEY=your_claude_key")
    print("     - GOOGLE_API_KEY=your_gemini_key")
    print("  2. Enter them when prompted")
    print("  3. Edit this script to hardcode them (not recommended)")

    # Check for environment variables
    claude_key = os.environ.get('ANTHROPIC_API_KEY')
    gemini_key = os.environ.get('GOOGLE_API_KEY')

    # Prompt for keys if not found
    if not claude_key:
        print("\n📝 Claude API Key not found in environment.")
        response = input("Enter Claude API key (or press Enter to skip): ").strip()
        if response:
            claude_key = response

    if not gemini_key:
        print("\n📝 Gemini API Key not found in environment.")
        response = input("Enter Gemini API key (or press Enter to skip): ").strip()
        if response:
            gemini_key = response

    if not claude_key and not gemini_key:
        print("\n⚠️  No API keys provided. Running without AI assistance.")
        print("You can still train normally, but won't get AI optimizations.")
        response = input("\nContinue without AI? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting. Please set up API keys and try again.")
            return

    # Setup AI training configuration
    ai_config = setup_ai_training(
        claude_api_key=claude_key,
        gemini_api_key=gemini_key
    )

    # Step 2: Load training data
    print("\n[Step 2] Loading training data...")
    data_dir = "sample_data"

    if not os.path.exists(data_dir):
        print(f"⚠️  Directory '{data_dir}' not found.")
        print("Creating sample data...")
        loader = DatasetLoader(data_dir)
        loader.create_sample_dataset(data_dir, num_samples=50)

    loader = DatasetLoader(data_dir)
    loader.scan_directory()
    data = loader.load_all_data(max_text=30, max_code=20)

    print(f"✅ Loaded {len(data['text'])} text files")

    # Step 3: Setup graph and processor
    print("\n[Step 3] Setting up neural network...")
    graph = GraphNetwork(name="ai_enhanced_graph")
    text_processor = TextProcessor(use_spacy=False, min_word_frequency=2)

    # Process training data
    print("\n[Step 4] Processing training data...")
    for i, text in enumerate(data['text']):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(data['text'])}...")
        text_processor.process_text(text, graph)

    print(f"✅ Graph constructed: {graph}")

    # Extract sequences
    print("\n[Step 5] Extracting training sequences...")
    sequences = []
    for text in data['text']:
        tokens = text_processor.tokenize(text)
        # Create sequences of length 10-15
        for i in range(0, len(tokens) - 10, 5):
            sequences.append(tokens[i:i+10])

    print(f"✅ Extracted {len(sequences)} training sequences")

    # Step 6: Train with AI assistance
    print("\n[Step 6] Training with AI assistance...")

    ai_trainer = AIEnhancedTrainingEngine(
        graph=graph,
        ai_config=ai_config
    )

    # Run AI-enhanced training
    metrics = ai_trainer.train_with_ai_assistance(sequences)

    # Step 7: Display results
    print("\n[Step 7] Training Results")
    print("="*70)

    if metrics:
        final = metrics[-1]
        print(f"Final Loss: {final.loss:.4f}")
        print(f"Final Avg Weight: {final.avg_weight:.4f}")
        print(f"Total Updates: {final.num_updates}")
        print(f"Training Time: {sum(m.time_elapsed for m in metrics):.2f}s")

    stats = graph.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Avg Edge Weight: {stats['avg_edge_weight']:.4f}")
    print(f"  Avg Node Degree: {stats['avg_node_degree']:.2f}")

    # Step 8: Save the model
    print("\n[Step 8] Saving model...")
    os.makedirs('ai_trained_models', exist_ok=True)
    graph.save('ai_trained_models/ai_enhanced_graph.pkl', format='pickle')
    print("✅ Model saved to: ai_trained_models/ai_enhanced_graph.pkl")

    # Step 9: Test the trained model
    print("\n[Step 9] Testing trained model...")
    from core.traversal_engine import TraversalEngine, TraversalConfig, TraversalStrategy

    traversal_engine = TraversalEngine(
        graph,
        TraversalConfig(
            strategy=TraversalStrategy.TEMPERATURE,
            max_tokens=50,
            temperature=0.8
        )
    )

    # Find a good starting token
    common_words = ['the', 'a', 'in', 'to', 'of', 'and']
    start_token = None
    for word in common_words:
        for node in graph.nodes.values():
            if node.value.lower() == word:
                start_token = word
                break
        if start_token:
            break

    if start_token:
        print(f"\nGenerating text starting with '{start_token}'...")
        generated = traversal_engine.generate_text(start_token)
        print(f"\nGenerated: {generated[:200]}...")

    print("\n" + "="*70)
    print("AI-ENHANCED TRAINING COMPLETE!")
    print("="*70)

    print("\n💡 Next Steps:")
    print("  1. Review AI suggestions above")
    print("  2. Upload more training data")
    print("  3. Train again with optimized parameters")
    print("  4. Use the model in your applications")

    print("\n📚 For more information:")
    print("  • AI Training Guide: AI_TRAINING_GUIDE.md")
    print("  • API Documentation: PROJECT_DOCUMENTATION.md")


def quick_ai_training():
    """Quick training with default settings"""
    print("🚀 Quick AI-Enhanced Training\n")

    # Use existing sample data
    loader = DatasetLoader("sample_data")
    data = loader.load_all_data(max_text=20)

    # Setup graph
    graph = GraphNetwork()
    processor = TextProcessor()

    # Process
    for text in data['text']:
        processor.process_text(text, graph)

    # Extract sequences
    sequences = []
    for text in data['text']:
        tokens = processor.tokenize(text)
        for i in range(0, len(tokens)-10, 5):
            sequences.append(tokens[i:i+10])

    # Train with AI
    ai_config = setup_ai_training()
    trainer = AIEnhancedTrainingEngine(graph, ai_config=ai_config)
    trainer.train_with_ai_assistance(sequences)

    print("\n✅ Quick training complete!")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_ai_training()
    else:
        main()

