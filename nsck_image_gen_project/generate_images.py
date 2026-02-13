#!/usr/bin/env python3
"""
NSCK Image Generation Inference
================================
Generate images from text descriptions using trained NSCK system.

NO NEURAL NETWORKS - Pure VSA/hypervector-based generation.

Usage:
  python generate_images.py "red circle"
  python generate_images.py "blue airplane" --size 128
  python generate_images.py --interactive
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

try:
    from python.core.multimodal.image_generator import ImageGenerator
    from python.core.memory.semantic_memory import SemanticMemory
    from python.core.memory.episodic_memory import EpisodicMemory
    from python.core.multimodal.multimodal_processor import MultimodalProcessor
    from python.core.reasoning.context_engine import ContextEngine
    print("[✓] NSCK modules imported successfully")
except ImportError as e:
    print(f"[✗] Import error: {e}")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[!] PIL not available, will only save as numpy array")


class ImageGeneratorApp:
    """Application for generating images from text."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize generator, optionally loading trained state."""
        print("Initializing NSCK Image Generator...")
        
        # Initialize cognitive systems
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory()
        self.context_engine = ContextEngine(self.semantic_memory)
        
        self.multimodal_processor = MultimodalProcessor(
            context_engine=self.context_engine,
            semantic_memory=self.semantic_memory
        )
        
        self.generator = ImageGenerator(
            semantic_memory=self.semantic_memory,
            multimodal_processor=self.multimodal_processor
        )
        
        # Load trained state if provided
        if model_path and os.path.exists(model_path):
            self.load_state(model_path)
        
        print("[✓] Generator ready")
    
    def load_state(self, model_path: str):
        """Load trained system state."""
        print(f"Loading trained state from {model_path}...")
        try:
            import pickle
            with open(model_path, 'rb') as f:
                state = pickle.load(f)
            
            if "semantic_memory" in state:
                self.semantic_memory = state["semantic_memory"]
                self.generator.semantic = self.semantic_memory
                print(f"  [✓] Loaded semantic memory")
            
            if "episodic_memory" in state:
                self.episodic_memory = state["episodic_memory"]
                print(f"  [✓] Loaded episodic memory")
            
            print("[✓] State loaded successfully")
        except Exception as e:
            print(f"[!] Warning: Could not load state: {e}")
            print("[!] Using untrained generator")
    
    def generate(
        self,
        prompt: str,
        size: Tuple[int, int] = (64, 64),
        output_path: Optional[str] = None
    ):
        """Generate an image from text prompt."""
        print(f"\nGenerating image for: '{prompt}'")
        print(f"Size: {size[0]}×{size[1]}")
        print(f"Using: VSA/Hypervectors (NO neural networks)")
        print("-" * 50)
        
        # Generate
        result = self.generator.generate(prompt, size=size)
        
        # Display info
        print(f"✓ Generated in {result.iterations} iterations")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Similarity to target: {result.similarity_to_target:.3f}")
        print(f"  Image shape: {result.image.shape}")
        print(f"  Features used:")
        print(f"    - Shape: {result.features_used.shape_hints}")
        print(f"    - Brightness: {result.features_used.brightness:.1f}")
        print(f"    - Edge density: {result.features_used.edge_density:.2f}")
        print(f"    - Dominant orientation: {result.features_used.dominant_orientation}")
        
        # Save if requested
        if output_path:
            self.save_image(result.image, output_path)
            print(f"✓ Saved to: {output_path}")
        
        return result
    
    def save_image(self, image_array: np.ndarray, path: str):
        """Save image to file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(exist_ok=True, parents=True)
        
        if HAS_PIL:
            # Save as PNG using PIL
            if image_array.ndim == 2:
                # Grayscale
                img = Image.fromarray(image_array, mode='L')
            else:
                # Color
                img = Image.fromarray(image_array, mode='RGB')
            img.save(path)
        else:
            # Save as numpy array
            np.save(path.replace('.png', '.npy'), image_array)
    
    def interactive(self):
        """Interactive mode - generate images from prompts."""
        print("\n" + "="*70)
        print("NSCK Image Generator - Interactive Mode")
        print("="*70)
        print("Enter text prompts to generate images.")
        print("Commands:")
        print("  - Type a description (e.g., 'red circle', 'blue airplane')")
        print("  - 'size WIDTHxHEIGHT' to change output size (e.g., 'size 128x128')")
        print("  - 'quit' or 'exit' to exit")
        print("="*70 + "\n")
        
        current_size = (64, 64)
        output_dir = Path("results/generated_images")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        image_count = 0
        
        while True:
            try:
                prompt = input("\n> ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if prompt.lower().startswith('size '):
                    # Parse size command
                    try:
                        size_str = prompt[5:].strip()
                        if 'x' in size_str:
                            w, h = size_str.split('x')
                            current_size = (int(h), int(w))
                            print(f"✓ Output size set to {current_size[0]}×{current_size[1]}")
                        else:
                            print("! Invalid format. Use: size WIDTHxHEIGHT (e.g., size 128x128)")
                    except:
                        print("! Invalid size format")
                    continue
                
                # Generate image
                image_count += 1
                output_path = output_dir / f"image_{image_count:03d}.png"
                
                result = self.generate(prompt, size=current_size, output_path=str(output_path))
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text using NSCK (VSA-based, no neural networks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single image
  python generate_images.py "red circle"
  
  # Generate with custom size
  python generate_images.py "blue airplane" --size 128
  
  # Generate and save
  python generate_images.py "green tree" --output results/tree.png
  
  # Interactive mode
  python generate_images.py --interactive
  
  # Use trained model
  python generate_images.py "cat" --model models/image_gen_system.pkl
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text description for image generation"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Image size (creates square image, default: 64)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for generated image"
    )
    
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to trained model (optional)"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Create app
    app = ImageGeneratorApp(model_path=args.model)
    
    # Run
    if args.interactive:
        app.interactive()
    elif args.prompt:
        size = (args.size, args.size)
        output = args.output or f"results/generated_{args.prompt.replace(' ', '_')[:30]}.png"
        app.generate(args.prompt, size=size, output_path=output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
