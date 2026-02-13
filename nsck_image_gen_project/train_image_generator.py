#!/usr/bin/env python3
"""
NSCK Image Generation Training Pipeline
========================================
Trains the NSCK image generation system on text-image pairs from CIFAR-10.
Learns associations between text descriptions and visual features using VSA.

NO NEURAL NETWORKS OR LLMs - Uses pure VSA/hypervector architecture.

Run: python train_image_generator.py --samples 500
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

# NSCK imports
try:
    from python.utils.hf_adapter import HuggingFaceAdapter
    from python.core.language.text_knowledge_learner import TextKnowledgeLearner
    from python.core.memory.semantic_memory import SemanticMemory
    from python.core.memory.episodic_memory import EpisodicMemory
    from python.core.multimodal.multimodal_processor import MultimodalProcessor, MultimodalInput
    from python.core.multimodal.image_generator import ImageGenerator
    from python.core.reasoning.context_engine import ContextEngine
    print("[✓] All NSCK modules imported successfully")
except ImportError as e:
    print(f"[✗] Import error: {e}")
    print("[!] Make sure you're running from the correct directory")
    sys.exit(1)


class TrainingMonitor:
    """Tracks training progress and logs events."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.output_dir / "training.log"
        self.metrics_file = self.output_dir / "metrics.json"
        
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "samples_processed": 0,
            "associations_learned": 0,
            "training_time": 0.0,
            "phases": []
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().isoformat()
        formatted = f"[{timestamp}] [{level:8s}] {message}"
        print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def record_metric(self, key: str, value: Any):
        """Record a metric."""
        self.metrics[key] = value
        self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class ImageGenTrainer:
    """Core training orchestration for image generation."""
    
    # CIFAR-10 labels
    CIFAR10_LABELS = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
        
        # Initialize cognitive systems
        self.monitor.log("Initializing NSCK cognitive systems...", "INFO")
        
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory()
        self.context_engine = ContextEngine(self.semantic_memory)
        
        self.multimodal_processor = MultimodalProcessor(
            context_engine=self.context_engine,
            semantic_memory=self.semantic_memory
        )
        
        self.text_learner = TextKnowledgeLearner(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            context_engine=self.context_engine
        )
        
        self.image_generator = ImageGenerator(
            semantic_memory=self.semantic_memory,
            multimodal_processor=self.multimodal_processor
        )
        
        # HuggingFace adapter for datasets
        self.adapter = HuggingFaceAdapter()
        
        self.monitor.log("[✓] All systems initialized", "SUCCESS")
    
    def train_on_image_text_pairs(self, num_samples: int = 500):
        """
        Train on image-text pairs from CIFAR-10.
        Learns associations between visual features and text descriptions.
        """
        self.monitor.log(f"\n{'='*60}", "INFO")
        self.monitor.log(f"TRAINING: Image-Text Association Learning", "START")
        self.monitor.log(f"Target samples: {num_samples}", "INFO")
        self.monitor.log(f"{'='*60}\n", "INFO")
        
        phase_start = time.time()
        phase_metrics = {
            "name": "image_text_association",
            "samples": num_samples,
            "processed": 0,
            "associations": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            count = 0
            for image_tensor, label_idx in self.adapter.stream_vision_data(
                dataset_name="cifar10",
                split="train",
                limit=num_samples
            ):
                try:
                    # Convert image
                    if hasattr(image_tensor, 'numpy'):
                        image_array = image_tensor.numpy()
                    else:
                        image_array = image_tensor
                    
                    # Ensure uint8 format
                    if image_array.dtype != np.uint8:
                        if image_array.max() <= 1.0:
                            image_array = (image_array * 255).astype(np.uint8)
                        else:
                            image_array = image_array.astype(np.uint8)
                    
                    # Get text label
                    label_text = self.CIFAR10_LABELS[label_idx]
                    
                    # Process image through multimodal processor
                    img_input = MultimodalInput(
                        image=image_array,
                        metadata={"label": label_text}
                    )
                    img_processed = self.multimodal_processor.process(img_input)
                    
                    # Process text description
                    text_input = MultimodalInput(
                        text=label_text,
                        metadata={"type": "label"}
                    )
                    text_processed = self.multimodal_processor.process(text_input)
                    
                    # Learn the association in semantic memory
                    # Store concept with both text and image HVs
                    concept_name = f"visual_{label_text}"
                    
                    # Bundle text and image HVs to create association
                    combined_hv = text_processed.fused_hv.bundle(img_processed.fused_hv)
                    
                    # Store in semantic memory
                    self.semantic_memory.add_concept(
                        concept_name,
                        combined_hv
                    )
                    
                    # Also learn descriptive text
                    description = f"a {label_text} with visual features"
                    self.text_learner.learn_from_text(description, f"train_{count}")
                    
                    count += 1
                    phase_metrics["processed"] += 1
                    phase_metrics["associations"] += 1
                    
                    if count % 50 == 0:
                        self.monitor.log(
                            f"[TRAIN] Processed {count}/{num_samples} pairs | "
                            f"Associations: {phase_metrics['associations']}",
                            "PROGRESS"
                        )
                    
                    if count >= num_samples:
                        break
                        
                except Exception as e:
                    self.monitor.log(f"Error processing sample {count}: {e}", "WARN")
                    continue
            
            phase_elapsed = time.time() - phase_start
            phase_metrics["duration"] = phase_elapsed
            phase_metrics["end_time"] = datetime.now().isoformat()
            
            self.monitor.metrics["samples_processed"] = count
            self.monitor.metrics["associations_learned"] = phase_metrics["associations"]
            self.monitor.metrics["training_time"] = phase_elapsed
            self.monitor.metrics["phases"].append(phase_metrics)
            
            self.monitor.log(
                f"\n[✓] TRAINING COMPLETE in {phase_elapsed:.1f}s\n"
                f"    Samples processed: {count}\n"
                f"    Associations learned: {phase_metrics['associations']}",
                "SUCCESS"
            )
            
        except Exception as e:
            self.monitor.log(f"TRAINING FAILED: {e}", "ERROR")
            phase_metrics["error"] = str(e)
            self.monitor.metrics["phases"].append(phase_metrics)
    
    def save_state(self, output_path: str = "models/image_gen_system.pkl"):
        """Save trained system state."""
        self.monitor.log(f"\nSaving trained system state to {output_path}...", "INFO")
        try:
            import pickle
            
            state = {
                "semantic_memory": self.semantic_memory,
                "episodic_memory": self.episodic_memory,
                "text_learner": self.text_learner,
                "image_generator": self.image_generator,
                "timestamp": datetime.now().isoformat(),
            }
            
            output_file = self.monitor.output_dir.parent / output_path
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'wb') as f:
                pickle.dump(state, f)
            
            self.monitor.log(f"[✓] System state saved to {output_file}", "SUCCESS")
            self.monitor.record_metric("model_path", str(output_file))
            
        except Exception as e:
            self.monitor.log(f"Failed to save state: {e}", "ERROR")


def main():
    parser = argparse.ArgumentParser(
        description="Train NSCK Image Generator on CIFAR-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python train_image_generator.py --samples 50
  
  # Full training
  python train_image_generator.py --samples 500
        """
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of image-text pairs to train on (default: 100)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs and metrics (default: logs)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("NSCK IMAGE GENERATION TRAINING")
    print("="*70)
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output_dir}")
    print(f"Architecture: VSA/Hypervectors (NO neural networks)")
    print("="*70 + "\n")
    
    # Create monitor
    monitor = TrainingMonitor(args.output_dir)
    monitor.log("Starting NSCK Image Generation Training", "START")
    monitor.log(f"Samples: {args.samples}", "INFO")
    
    # Create trainer
    trainer = ImageGenTrainer(monitor)
    
    # Train
    trainer.train_on_image_text_pairs(args.samples)
    
    # Save
    trainer.save_state()
    
    # Final summary
    monitor.log("\n" + "="*70, "INFO")
    monitor.log("TRAINING COMPLETE", "SUCCESS")
    monitor.log(f"Total time: {monitor.metrics['training_time']:.1f}s", "INFO")
    monitor.log(f"Samples processed: {monitor.metrics['samples_processed']}", "INFO")
    monitor.log(f"Associations learned: {monitor.metrics['associations_learned']}", "INFO")
    monitor.log(f"Logs saved to: {monitor.log_file}", "INFO")
    monitor.log(f"Metrics saved to: {monitor.metrics_file}", "INFO")
    monitor.log("="*70, "INFO")


if __name__ == "__main__":
    main()
