#!/usr/bin/env python3
"""
NSCK Multimodal Training Pipeline
===================================
Trains the NSCK cognitive system on real HuggingFace data:
- English text from WikiText-2
- Images from CIFAR-10

Provides real-time monitoring and saves training artifacts.

Run: python train_multimodal.py --text-samples 500 --image-samples 500
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
    from python.core.reasoning.cognitive_engine import CognitiveEngine
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
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = self.output_dir / "training.log"
        self.metrics_file = self.output_dir / "metrics.json"
        
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "text_samples": 0,
            "image_samples": 0,
            "concepts_learned": 0,
            "relations_learned": 0,
            "facts_stored": 0,
            "errors": 0,
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


class MultimodalTrainer:
    """Core training orchestration."""
    
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
        
        # Optional: Initialize cognitive engine for reasoning
        try:
            self.cognitive_engine = CognitiveEngine(
                semantic_memory=self.semantic_memory,
                episodic_memory=self.episodic_memory,
                world_model_type="basic"
            )
            self.monitor.log("CognitiveEngine initialized", "INFO")
        except Exception as e:
            self.monitor.log(f"CognitiveEngine init failed (optional): {e}", "WARN")
            self.cognitive_engine = None
        
        self.adapter = HuggingFaceAdapter()
        self.monitor.log("All systems ready", "SUCCESS")
    
    def train_on_text(self, num_samples: int = 100):
        """Train on English text from WikiText-2."""
        self.monitor.log(f"\n{'='*60}", "INFO")
        self.monitor.log(f"PHASE 1: TEXT LEARNING (WikiText-2)", "START")
        self.monitor.log(f"Target samples: {num_samples}", "INFO")
        self.monitor.log(f"{'='*60}\n", "INFO")
        
        phase_start = time.time()
        phase_metrics = {
            "name": "text_learning",
            "samples": num_samples,
            "concepts": 0,
            "relations": 0,
            "facts": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            count = 0
            for text in self.adapter.stream_text_data(
                dataset_name="wikitext",
                config="wikitext-2-v1",
                split="train",
                limit=num_samples
            ):
                try:
                    # Filter empty/short text
                    if len(text.strip()) < 50:
                        continue
                    
                    # Learn from text
                    result = self.text_learner.learn_from_text(text)
                    
                    # Handle both dict and LearningSession returns
                    if isinstance(result, dict):
                        stats = result
                        concepts = stats.get('concepts', 0)
                        relations = stats.get('relations', 0)
                        facts = stats.get('facts', 0)
                    else:
                        # LearningSession object
                        concepts = result.concepts_learned if hasattr(result, 'concepts_learned') else 0
                        relations = result.relations_learned if hasattr(result, 'relations_learned') else 0
                        facts = result.facts_stored if hasattr(result, 'facts_stored') else 0
                    
                    count += 1
                    phase_metrics["concepts"] += concepts
                    phase_metrics["relations"] += relations
                    phase_metrics["facts"] += facts
                    
                    if count % 10 == 0:
                        self.monitor.log(
                            f"[TEXT] Processed {count}/{num_samples} samples | "
                            f"Concepts: {phase_metrics['concepts']} | "
                            f"Relations: {phase_metrics['relations']}",
                            "PROGRESS"
                        )
                    
                    if count >= num_samples:
                        break
                        
                except Exception as e:
                    phase_metrics["errors"] += 1
                    self.monitor.log(f"Error processing text sample {count}: {e}", "WARN")
                    continue
            
            phase_elapsed = time.time() - phase_start
            phase_metrics["duration"] = phase_elapsed
            phase_metrics["end_time"] = datetime.now().isoformat()
            
            self.monitor.metrics["text_samples"] = count
            self.monitor.metrics["concepts_learned"] = phase_metrics["concepts"]
            self.monitor.metrics["relations_learned"] = phase_metrics["relations"]
            self.monitor.metrics["facts_stored"] = phase_metrics["facts"]
            self.monitor.metrics["phases"].append(phase_metrics)
            
            self.monitor.log(
                f"\n[✓] TEXT PHASE COMPLETE in {phase_elapsed:.1f}s\n"
                f"    Samples processed: {count}\n"
                f"    Concepts learned: {phase_metrics['concepts']}\n"
                f"    Relations learned: {phase_metrics['relations']}\n"
                f"    Facts stored: {phase_metrics['facts']}\n"
                f"    Errors: {phase_metrics['errors']}",
                "SUCCESS"
            )
            
        except Exception as e:
            self.monitor.log(f"TEXT PHASE FAILED: {e}", "ERROR")
            phase_metrics["error"] = str(e)
            self.monitor.metrics["phases"].append(phase_metrics)
    
    def train_on_images(self, num_samples: int = 100):
        """Train on images from CIFAR-10."""
        self.monitor.log(f"\n{'='*60}", "INFO")
        self.monitor.log(f"PHASE 2: IMAGE LEARNING (CIFAR-10)", "START")
        self.monitor.log(f"Target samples: {num_samples}", "INFO")
        self.monitor.log(f"{'='*60}\n", "INFO")
        
        phase_start = time.time()
        phase_metrics = {
            "name": "image_learning",
            "samples": num_samples,
            "processed": 0,
            "features_extracted": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            count = 0
            for image_tensor, label in self.adapter.stream_vision_data(
                dataset_name="cifar10",
                split="train",
                limit=num_samples
            ):
                try:
                    # Convert torch tensor to numpy if needed
                    if hasattr(image_tensor, 'numpy'):
                        image_array = image_tensor.numpy()
                    else:
                        image_array = image_tensor
                    
                    # Ensure uint8 format (0-255)
                    if image_array.dtype != np.uint8:
                        if image_array.max() <= 1.0:
                            image_array = (image_array * 255).astype(np.uint8)
                        else:
                            image_array = image_array.astype(np.uint8)
                    
                    # Process through multimodal processor
                    multimodal_input = MultimodalInput(
                        image=image_array,
                        metadata={"label": label, "dataset": "cifar10"}
                    )
                    
                    processed = self.multimodal_processor.process(multimodal_input)
                    
                    # Store in episodic memory
                    # (Create minimal episode for image)
                    episode_state = {
                        "image_label": label,
                        "extracted_concepts": processed.extracted_concepts
                    }
                    
                    count += 1
                    phase_metrics["processed"] += 1
                    phase_metrics["features_extracted"] += len(processed.extracted_concepts)
                    
                    if count % 20 == 0:
                        self.monitor.log(
                            f"[IMAGE] Processed {count}/{num_samples} images | "
                            f"Features extracted: {phase_metrics['features_extracted']}",
                            "PROGRESS"
                        )
                    
                    if count >= num_samples:
                        break
                        
                except Exception as e:
                    phase_metrics["errors"] += 1
                    self.monitor.log(f"Error processing image {count}: {e}", "WARN")
                    continue
            
            phase_elapsed = time.time() - phase_start
            phase_metrics["duration"] = phase_elapsed
            phase_metrics["end_time"] = datetime.now().isoformat()
            
            self.monitor.metrics["image_samples"] = count
            self.monitor.metrics["phases"].append(phase_metrics)
            
            self.monitor.log(
                f"\n[✓] IMAGE PHASE COMPLETE in {phase_elapsed:.1f}s\n"
                f"    Samples processed: {count}\n"
                f"    Features extracted: {phase_metrics['features_extracted']}\n"
                f"    Errors: {phase_metrics['errors']}",
                "SUCCESS"
            )
            
        except Exception as e:
            self.monitor.log(f"IMAGE PHASE FAILED: {e}", "ERROR")
            phase_metrics["error"] = str(e)
            self.monitor.metrics["phases"].append(phase_metrics)
    
    def save_state(self, output_path: str = "models/trained_system.pkl"):
        """Save trained system state."""
        self.monitor.log(f"\nSaving trained system state to {output_path}...", "INFO")
        try:
            import pickle
            
            state = {
                "semantic_memory": self.semantic_memory,
                "episodic_memory": self.episodic_memory,
                "text_learner": self.text_learner,
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
        description="Train NSCK on HuggingFace data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small datasets
  python train_multimodal.py --text-samples 50 --image-samples 50
  
  # Full training
  python train_multimodal.py --text-samples 500 --image-samples 500
  
  # Only text
  python train_multimodal.py --text-samples 250 --skip-images
        """
    )
    
    parser.add_argument(
        "--text-samples",
        type=int,
        default=100,
        help="Number of text samples to train on (default: 100)"
    )
    parser.add_argument(
        "--image-samples",
        type=int,
        default=100,
        help="Number of image samples to train on (default: 100)"
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text learning phase"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image learning phase"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs and results"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    monitor = TrainingMonitor(output_dir)
    
    # Banner
    monitor.log("\n" + "="*60, "INFO")
    monitor.log("NSCK MULTIMODAL TRAINING PIPELINE", "INFO")
    monitor.log("="*60, "INFO")
    monitor.log(f"Started at {datetime.now().isoformat()}", "INFO")
    monitor.log(f"Output directory: {output_dir}", "INFO")
    monitor.log("="*60 + "\n", "INFO")
    
    # Initialize trainer
    trainer = MultimodalTrainer(monitor)
    
    # Run training phases
    overall_start = time.time()
    
    if not args.skip_text:
        trainer.train_on_text(args.text_samples)
    
    if not args.skip_images:
        trainer.train_on_images(args.image_samples)
    
    overall_elapsed = time.time() - overall_start
    
    # Save results
    monitor.log("\n" + "="*60, "INFO")
    monitor.log("SAVING TRAINED SYSTEM STATE", "INFO")
    monitor.log("="*60 + "\n", "INFO")
    
    trainer.save_state()
    
    # Final summary
    monitor.metrics["training_time"] = overall_elapsed
    monitor.metrics["end_time"] = datetime.now().isoformat()
    monitor.save_metrics()
    
    monitor.log("\n" + "="*60, "SUCCESS")
    monitor.log("TRAINING COMPLETE", "SUCCESS")
    monitor.log("="*60, "SUCCESS")
    monitor.log(f"Total training time: {overall_elapsed:.1f}s", "SUCCESS")
    monitor.log(f"Logs saved to: {monitor.log_file}", "SUCCESS")
    monitor.log(f"Metrics saved to: {monitor.metrics_file}", "SUCCESS")
    monitor.log("="*60 + "\n", "SUCCESS")
    
    print("\n[Next Steps]")
    print("1. Review logs in: " + str(monitor.log_file))
    print("2. Run tests with: python test_trained_system.py")
    print("3. Chat with trained model: python chat_with_trained_model.py")


if __name__ == "__main__":
    main()
