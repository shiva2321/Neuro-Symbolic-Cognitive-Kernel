
import time
import os
import sys
from typing import Optional, Dict
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.tools.dataset_streamer import TextStreamer
from python.core.memory.semantic_memory import SemanticMemory

class VSATrainer:
    """
    Orchestrates the training of the VSA Language Model using streaming data.
    """
    
    def __init__(self, memory_path: str = "data/vsa_memory.pkl"):
        self.memory_path = memory_path
        
        # Initialize Memory (Load if exists)
        self.memory = SemanticMemory()
        if os.path.exists(self.memory_path):
            print(f"[Trainer] Loading existing memory from {self.memory_path}...")
            try:
                self.memory.load(self.memory_path)
                print(f"[Trainer] Loaded {len(self.memory.concept_hvs)} concepts.")
            except Exception as e:
                print(f"[Trainer] Failed to load memory: {e}. Starting fresh.")
        
        # Initialize Learner
        self.learner = TextKnowledgeLearner(semantic_memory=self.memory)
        
        # Metrics
        self.metrics = {
            "sentences_processed": 0,
            "concepts_learned": 0,
            "relations_learned": 0,
            "start_time": 0.0,
            "batches_processed": 0
        }
        
    def train(self, dataset_name: str = "wikitext", steps: int = 100, batch_size: int = 10):
        """
        Run the training loop.
        Args:
            dataset_name: Name of dataset to stream (wikitext, tinystories)
            steps: Number of SENTENCES to process (approx)
            batch_size: Sentences per batch
        """
        print(f"\n[Trainer] Starting training on '{dataset_name}' for {steps} steps...")
        
        streamer = TextStreamer(dataset_name)
        streamer.load()
        
        self.metrics["start_time"] = time.time()
        
        try:
            for batch in streamer.stream_batches(batch_size=batch_size, limit=steps):
                # Join batch into a single text block for the learner (optimization)
                # properly separated by periods if needed, but learner splits sentences anyway.
                text_block = " ".join(batch)
                
                # Run Learner
                # We use a dummy source for stream
                stats = self.learner.learn_from_text(
                    text=text_block, 
                    source=f"hf_{dataset_name}"
                )
                
                # Update Metrics
                self.metrics["sentences_processed"] += len(batch)
                self.metrics["concepts_learned"] += stats.get("concepts", 0)
                self.metrics["relations_learned"] += stats.get("relations", 0)
                self.metrics["batches_processed"] += 1
                
                # Log Progress
                self._log_progress()
                
                # Checkpoint every 50 batches
                if self.metrics["batches_processed"] % 50 == 0:
                    self.save_checkpoint()
                    
        except KeyboardInterrupt:
            print("\n[Trainer] Training interrupted by user.")
        except Exception as e:
            print(f"\n[Trainer] Error during training: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.save_checkpoint()
            self._print_summary()
            
    def save_checkpoint(self):
        """Save the semantic memory to disk."""
        print(f" [Checkpoint] Saving memory to {self.memory_path}...")
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self.memory.save(self.memory_path)
        
    def _log_progress(self):
        """Print inline progress stats."""
        elapsed = time.time() - self.metrics["start_time"]
        sps = self.metrics["sentences_processed"] / (elapsed + 0.001)
        
        sys.stdout.write(
            f"\r[Training] Sentences: {self.metrics['sentences_processed']} | "
            f"Concepts: {self.metrics['concepts_learned']} | "
            f"Relations: {self.metrics['relations_learned']} | "
            f"Speed: {sps:.1f} sent/s"
        )
        sys.stdout.flush()
        
    def _print_summary(self):
        """Print final stats."""
        elapsed = time.time() - self.metrics["start_time"]
        print(f"\n\n{'-'*40}")
        print(f"Training Complete in {elapsed:.2f}s")
        print(f"Total Sentences: {self.metrics['sentences_processed']}")
        print(f"Total Concepts:  {self.metrics['concepts_learned']}")
        print(f"Total Relations: {self.metrics['relations_learned']}")
        print(f"{'-'*40}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train NSCK VSA Language Model")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to stream (wikitext, tinystories)")
    parser.add_argument("--steps", type=int, default=100, help="Number of sentences to process")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--memory", type=str, default="data/vsa_memory.pkl", help="Path to semantic memory file")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.memory), exist_ok=True)
    
    trainer = VSATrainer(memory_path=args.memory)
    trainer.train(dataset_name=args.dataset, steps=args.steps, batch_size=args.batch_size)
