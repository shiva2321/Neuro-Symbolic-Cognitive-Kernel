"""
NSCK Image Generation Training Pipeline
========================================
Trains the VSA-based image generator on text-image pairs from HuggingFace datasets.

NO NEURAL NETWORKS - Uses:
- VSA hypervectors for representation
- Semantic Folding for text encoding
- Classical computer vision for image features
- Associative memory for concept-visual mappings

Dataset: CIFAR-10 (with class labels as text descriptions)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from typing import List, Tuple, Optional
import pickle
from pathlib import Path

from python.core.multimodal.image_generator import ImageGenerator, GenerationConfig, VisualFeatures
from python.core.memory.semantic_memory import SemanticMemory
from python.utils.hf_adapter import HuggingFaceAdapter


# CIFAR-10 class labels
CIFAR10_LABELS = {
    0: "airplane flying in sky",
    1: "automobile car vehicle",
    2: "bird with wings",
    3: "cat with fur",
    4: "deer with antlers",
    5: "dog with tail",
    6: "frog amphibian green",
    7: "horse with mane",
    8: "ship boat water",
    9: "truck vehicle wheels"
}


class ImageGenerationTrainer:
    """
    Trainer for the VSA-based image generator.
    """
    
    def __init__(
        self,
        output_dir: str = "./models/image_generation",
        use_color: bool = True,
        image_size: Tuple[int, int] = (32, 32)
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create generator with config
        config = GenerationConfig(
            image_size=image_size,
            is_color=use_color,
            smoothing_factor=0.3,
            contrast_boost=1.1
        )
        
        self.generator = ImageGenerator(
            semantic_memory=SemanticMemory(),
            config=config
        )
        
        # HuggingFace adapter for data
        self.hf_adapter = HuggingFaceAdapter()
        
        # Training statistics
        self.train_stats = {
            "examples_processed": 0,
            "concepts_learned": 0,
            "training_time": 0.0
        }
    
    def prepare_training_data(
        self,
        dataset_name: str = "cifar10",
        split: str = "train",
        max_examples: int = 1000
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Prepare training data from HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset (default: cifar10)
            split: Dataset split to use (train/test)
            max_examples: Maximum number of examples to load
            
        Returns:
            List of (text_description, image_array) tuples
        """
        print(f"Loading {dataset_name} dataset (split={split}, max={max_examples})...")
        
        data_pairs = []
        
        try:
            # Stream data from HuggingFace
            for idx, (img_tensor, label) in enumerate(self.hf_adapter.stream_vision_data(
                dataset_name=dataset_name,
                split=split,
                limit=max_examples
            )):
                # Convert tensor to numpy (C, H, W) -> (H, W, C)
                if hasattr(img_tensor, 'numpy'):
                    img_np = img_tensor.numpy()
                else:
                    img_np = np.array(img_tensor)
                
                # Ensure proper shape
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                    # Channel first -> channel last
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Convert to uint8 [0, 255]
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    img_np = (img_np * 255).astype(np.uint8)
                
                # Get text description
                text = CIFAR10_LABELS.get(label, f"object class {label}")
                
                data_pairs.append((text, img_np))
                
                if (idx + 1) % 100 == 0:
                    print(f"  Loaded {idx + 1}/{max_examples} examples")
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Generating synthetic training data instead...")
            data_pairs = self._generate_synthetic_data(max_examples)
        
        print(f"Loaded {len(data_pairs)} training examples")
        return data_pairs
    
    def _generate_synthetic_data(self, num_examples: int = 100) -> List[Tuple[str, np.ndarray]]:
        """Generate synthetic training data for testing."""
        print("Generating synthetic training data...")
        
        data_pairs = []
        rng = np.random.RandomState(42)
        
        concepts = [
            ("red circle", [255, 0, 0]),
            ("blue square", [0, 0, 255]),
            ("green triangle", [0, 255, 0]),
            ("yellow star", [255, 255, 0]),
            ("purple oval", [128, 0, 128]),
            ("orange rectangle", [255, 165, 0]),
            ("pink heart", [255, 192, 203]),
            ("cyan diamond", [0, 255, 255])
        ]
        
        for i in range(num_examples):
            text, color = concepts[i % len(concepts)]
            
            # Generate simple colored image
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            
            # Fill with color
            base_color = np.array(color, dtype=np.uint8)
            for ch in range(3):
                img[:, :, ch] = base_color[ch] + rng.randint(-30, 30, (32, 32))
            
            img = np.clip(img, 0, 255)
            
            data_pairs.append((text, img))
        
        return data_pairs
    
    def train(
        self,
        dataset_name: str = "cifar10",
        split: str = "train",
        max_examples: int = 1000,
        save_interval: int = 200
    ):
        """
        Train the image generator.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split
            max_examples: Maximum training examples
            save_interval: Save checkpoint every N examples
        """
        import time
        
        print("=" * 60)
        print("NSCK Image Generation Training")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Max examples: {max_examples}")
        print(f"Output dir: {self.output_dir}")
        print()
        
        # Load training data
        start_time = time.time()
        train_data = self.prepare_training_data(dataset_name, split, max_examples)
        
        if not train_data:
            print("ERROR: No training data available!")
            return
        
        # Train the generator
        print("\nStarting training...")
        self.generator.train_from_examples(train_data, max_examples=max_examples)
        
        elapsed_time = time.time() - start_time
        
        # Update statistics
        self.train_stats["examples_processed"] = len(train_data)
        self.train_stats["concepts_learned"] = len(self.generator.concept_memory.feature_templates)
        self.train_stats["training_time"] = elapsed_time
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Examples processed: {self.train_stats['examples_processed']}")
        print(f"Concepts learned: {self.train_stats['concepts_learned']}")
        print(f"Training time: {elapsed_time:.2f}s")
        print()
        
        # Save the trained model
        self.save_model()
    
    def save_model(self, filename: Optional[str] = None):
        """Save the trained generator model."""
        if filename is None:
            filename = "image_generator_model.pkl"
        
        save_path = self.output_dir / filename
        
        print(f"Saving model to {save_path}...")
        
        # Save generator state
        model_state = {
            "concept_memory": self.generator.concept_memory,
            "config": self.generator.config,
            "stats": self.generator.get_statistics(),
            "train_stats": self.train_stats
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model saved successfully!")
        
        # Also save a text summary
        summary_path = self.output_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("NSCK Image Generation Model - Training Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Examples processed: {self.train_stats['examples_processed']}\n")
            f.write(f"Concepts learned: {self.train_stats['concepts_learned']}\n")
            f.write(f"Training time: {self.train_stats['training_time']:.2f}s\n\n")
            f.write("Learned concepts:\n")
            for concept in self.generator.get_statistics()['concept_list']:
                f.write(f"  - {concept}\n")
        
        print(f"Training summary saved to {summary_path}")
    
    def load_model(self, filename: Optional[str] = None):
        """Load a trained generator model."""
        if filename is None:
            filename = "image_generator_model.pkl"
        
        load_path = self.output_dir / filename
        
        if not load_path.exists():
            print(f"Model file not found: {load_path}")
            return False
        
        print(f"Loading model from {load_path}...")
        
        try:
            with open(load_path, 'rb') as f:
                model_state = pickle.load(f)
            
            self.generator.concept_memory = model_state["concept_memory"]
            self.generator.config = model_state["config"]
            self.train_stats = model_state.get("train_stats", {})
            
            print("Model loaded successfully!")
            print(f"Learned concepts: {len(self.generator.concept_memory.feature_templates)}")
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def test_generation(self, prompts: Optional[List[str]] = None):
        """
        Test the generator with sample prompts.
        
        Args:
            prompts: List of text prompts to test
        """
        if prompts is None:
            prompts = [
                "red airplane flying",
                "blue car automobile",
                "green bird with wings",
                "brown cat with fur",
                "gray ship on water"
            ]
        
        print("\n" + "=" * 60)
        print("Testing Image Generation")
        print("=" * 60)
        
        for idx, prompt in enumerate(prompts):
            print(f"\n[{idx+1}/{len(prompts)}] Generating: '{prompt}'")
            
            try:
                image = self.generator.generate(prompt)
                
                # Save generated image
                save_path = self.output_dir / f"generated_{idx+1}.npy"
                np.save(save_path, image)
                
                print(f"  ✓ Generated image: {image.shape}, dtype={image.dtype}")
                print(f"    Saved to: {save_path}")
                
                # Try to save as PNG if PIL is available
                try:
                    from PIL import Image as PILImage
                    png_path = self.output_dir / f"generated_{idx+1}.png"
                    
                    if image.ndim == 3:
                        pil_img = PILImage.fromarray(image, mode='RGB')
                    else:
                        pil_img = PILImage.fromarray(image, mode='L')
                    
                    pil_img.save(png_path)
                    print(f"    PNG saved to: {png_path}")
                
                except ImportError:
                    pass
            
            except Exception as e:
                print(f"  ✗ Error generating image: {e}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train NSCK Image Generator")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="HuggingFace dataset name (default: cifar10)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split (default: train)")
    parser.add_argument("--max-examples", type=int, default=1000,
                       help="Maximum training examples (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="./models/image_generation",
                       help="Output directory for models")
    parser.add_argument("--no-color", action="store_true",
                       help="Generate grayscale images")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test generation (load existing model)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic training data")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ImageGenerationTrainer(
        output_dir=args.output_dir,
        use_color=not args.no_color,
        image_size=(32, 32)
    )
    
    # Test only mode
    if args.test_only:
        if trainer.load_model():
            trainer.test_generation()
        else:
            print("Could not load model. Run training first.")
        return
    
    # Train the model
    if args.synthetic:
        # Use synthetic data
        print("Using synthetic training data...")
        train_data = trainer._generate_synthetic_data(args.max_examples)
        trainer.generator.train_from_examples(train_data)
        trainer.train_stats["examples_processed"] = len(train_data)
        trainer.train_stats["concepts_learned"] = len(trainer.generator.concept_memory.feature_templates)
        trainer.save_model()
    else:
        # Use HuggingFace dataset
        trainer.train(
            dataset_name=args.dataset,
            split=args.split,
            max_examples=args.max_examples
        )
    
    # Test generation
    trainer.test_generation()
    
    print("\n✓ Training and testing complete!")
    print(f"  Model saved to: {trainer.output_dir}")
    print(f"  Generated images saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
