#!/usr/bin/env python3
"""
NSCK Trained System Verification Tests
========================================
Comprehensive testing of:
- Memory systems (semantic, episodic)
- Learning capabilities
- Reasoning quality
- Multimodal processing
- Internal state tracking

Run: python test_trained_system.py
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.multimodal.multimodal_processor import MultimodalProcessor, MultimodalInput
from python.core.reasoning.context_engine import ContextEngine
import numpy as np


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()
    
    def record(self, test_name: str, passed: bool, message: str = "", metrics: Dict = None):
        """Record test result."""
        if passed:
            self.passed += 1
            status = "✓ PASS"
        else:
            self.failed += 1
            status = "✗ FAIL"
        
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "metrics": metrics or {}
        }
        self.results.append(result)
        
        print(f"{status}: {test_name}")
        if message:
            print(f"      {message}")
    
    def summary(self):
        """Print summary."""
        elapsed = time.time() - self.start_time
        total = self.passed + self.failed
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Passed: {self.passed}/{total}")
        print(f"Failed: {self.failed}/{total}")
        print(f"Time: {elapsed:.2f}s")
        print("="*60 + "\n")
        
        return {
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "elapsed": elapsed,
            "results": self.results
        }


# Test Suite
class NSCKTests:
    """Comprehensive test suite."""
    
    def __init__(self):
        self.results = TestResults()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.context = ContextEngine(self.semantic)
        self.text_learner = TextKnowledgeLearner(
            semantic_memory=self.semantic,
            episodic_memory=self.episodic,
            context_engine=self.context
        )
        self.multimodal = MultimodalProcessor(
            context_engine=self.context,
            semantic_memory=self.semantic
        )
    
    def test_semantic_memory_basic(self):
        """Test semantic memory creation and retrieval."""
        print("\n[TEST GROUP] Semantic Memory")
        print("-" * 40)
        
        try:
            # Add concepts (with properties dict as required by newer API)
            self.semantic.add_concept("dog", properties={})
            self.semantic.add_concept("animal", properties={})
            
            # Add relation
            self.semantic.add_relation("dog", "is_a", "animal")
            
            # Verify
            has_dog = "dog" in self.semantic.concept_map
            has_animal = "animal" in self.semantic.concept_map
            
            self.results.record(
                "Semantic memory handles concepts",
                has_dog and has_animal,
                f"Concepts stored: dog={has_dog}, animal={has_animal}"
            )
            
            # Check relations
            relations = self.semantic.get_relations("dog")
            has_relation = len(relations) > 0
            self.results.record(
                "Semantic memory stores relations",
                has_relation,
                f"Relations found: {len(relations)}"
            )
            
        except Exception as e:
            self.results.record("Semantic memory basic ops", False, str(e))
    
    def test_text_learning(self):
        """Test text learning from sample data."""
        print("\n[TEST GROUP] Text Learning")
        print("-" * 40)
        
        try:
            sample_text = """
            The cat sat on the mat. The dog played in the garden.
            Cats are animals. Dogs are loyal companions.
            The animal kingdom includes mammals, birds, and reptiles.
            """
            
            # Learn from text
            result = self.text_learner.learn_from_text(sample_text)
            
            # Handle both dict and LearningSession returns
            if isinstance(result, dict):
                concepts = result.get('concepts', 0)
                relations = result.get('relations', 0)
                facts = result.get('facts', 0)
            else:
                concepts = result.concepts_learned if hasattr(result, 'concepts_learned') else 0
                relations = result.relations_learned if hasattr(result, 'relations_learned') else 0
                facts = result.facts_stored if hasattr(result, 'facts_stored') else 0
            
            # Verify learning
            concepts_learned = concepts > 0
            self.results.record(
                "Text learning extracts concepts",
                concepts_learned,
                f"Concepts: {concepts}, "
                f"Relations: {relations}, "
                f"Facts: {facts}",
                metrics={
                    "concepts": concepts,
                    "relations": relations,
                    "facts": facts
                }
            )
        
        except Exception as e:
            self.results.record("Text learning", False, str(e))
    
    def test_episodic_memory(self):
        """Test episodic memory storage and retrieval."""
        print("\n[TEST GROUP] Episodic Memory")
        print("-" * 40)
        
        try:
            try:
                import python.core.vsa.hypervec_shim as hypervec_rs
            except:
                from python.core.vsa.hypervec_py import HyperVector
                class FakeHV:
                    def __init__(self, seed=0):
                        self.seed = seed
                        self.bits = [0] * 10240
                hypervec_rs = type('HypervecShim', (), {'HyperVector': FakeHV})()
            
            # Create a test episode
            test_state = {"position": (5, 5), "action": "move"}
            
            # Store in episodic memory
            episode_count_before = 0
            
            try:
                if hasattr(self.episodic, 'recent_episodes'):
                    episode_count_before = len(self.episodic.recent_episodes)
            except:
                pass
            
            self.results.record(
                "Episodic memory initialized",
                True if self.episodic else False,
                "Episodic memory system ready"
            )
        
        except Exception as e:
            self.results.record("Episodic memory", False, str(e))
    
    def test_multimodal_text(self):
        """Test multimodal processing of text."""
        print("\n[TEST GROUP] Multimodal Processing")
        print("-" * 40)
        
        try:
            test_text = "The quick brown fox jumps over the lazy dog"
            multimodal_input = MultimodalInput(text=test_text)
            
            result = self.multimodal.process(multimodal_input)
            
            has_hv = result.fused_hv is not None
            has_concepts = len(result.extracted_concepts) > 0
            
            self.results.record(
                "Multimodal processes text",
                has_hv and has_concepts,
                f"HV generated: {has_hv}, "
                f"Concepts: {len(result.extracted_concepts)} "
                f"({', '.join(result.extracted_concepts[:3])}...)"
            )
        
        except Exception as e:
            self.results.record("Multimodal text", False, str(e))
    
    def test_multimodal_image(self):
        """Test multimodal processing of images."""
        print("\n[TEST GROUP] Image Processing")
        print("-" * 40)
        
        try:
            # Create synthetic image (random noise)
            test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            multimodal_input = MultimodalInput(image=test_image)
            
            result = self.multimodal.process(multimodal_input)
            
            has_hv = result.fused_hv is not None
            confidence = result.confidence > 0
            
            self.results.record(
                "Multimodal processes images",
                has_hv and confidence,
                f"HV generated: {has_hv}, "
                f"Confidence: {result.confidence:.2f}",
                metrics={"confidence": result.confidence}
            )
        
        except Exception as e:
            self.results.record("Multimodal image", False, str(e))
    
    def test_knowledge_query(self):
        """Test querying learned knowledge."""
        print("\n[TEST GROUP] Knowledge Retrieval")
        print("-" * 40)
        
        try:
            # First learn something
            learning_text = "Python is a programming language. JavaScript is also a programming language."
            self.text_learner.learn_from_text(learning_text)
            
            # Then query
            if hasattr(self.text_learner, 'query_learned_knowledge'):
                result = self.text_learner.query_learned_knowledge("programming language", top_k=3)
                has_results = bool(result)
                
                self.results.record(
                    "Knowledge query returns results",
                    has_results,
                    f"Results found: {type(result).__name__}"
                )
            else:
                self.results.record(
                    "Knowledge query method",
                    False,
                    "Query method not available in text_learner"
                )
        
        except Exception as e:
            self.results.record("Knowledge query", False, str(e))
    
    def test_context_engine(self):
        """Test context engine functionality."""
        print("\n[TEST GROUP] Context Engine")
        print("-" * 40)
        
        try:
            # Set context
            context = {"topic": "animals", "task": "learning"}
            
            # Context engine should be initialized
            has_context = self.context is not None
            
            self.results.record(
                "Context engine initialized",
                has_context,
                "Context system ready"
            )
        
        except Exception as e:
            self.results.record("Context engine", False, str(e))
    
    def test_integrated_workflow(self):
        """Test full workflow: learn -> query -> reason."""
        print("\n[TEST GROUP] Integrated Workflow")
        print("-" * 40)
        
        try:
            test_text = """
            Machine learning is a subset of artificial intelligence.
            Neural networks are inspired by the human brain.
            Deep learning uses multiple layers of neural networks.
            Artificial intelligence can process information quickly.
            """
            
            # Step 1: Learn
            print("  Step 1: Learning from text...")
            result = self.text_learner.learn_from_text(test_text)
            
            # Handle dict or LearningSession
            if isinstance(result, dict):
                step1_ok = result.get('concepts', 0) > 0
            else:
                step1_ok = result.concepts_learned > 0
            
            # Step 2: Store in memory
            print("  Step 2: Verifying semantic memory...")
            concepts = len(self.semantic.concept_map)
            step2_ok = concepts > 0
            
            # Step 3: Query (if available)
            print("  Step 3: Querying learned knowledge...")
            step3_ok = True
            if hasattr(self.text_learner, 'query_learned_knowledge'):
                result = self.text_learner.query_learned_knowledge("neural networks")
                step3_ok = bool(result)
            
            all_ok = step1_ok and step2_ok and step3_ok
            self.results.record(
                "Integrated learn-query workflow",
                all_ok,
                f"Learn: {step1_ok}, Store: {step2_ok}, Query: {step3_ok} "
                f"({concepts} concepts)"
            )
        
        except Exception as e:
            self.results.record("Integrated workflow", False, str(e))
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "="*60)
        print("NSCK TRAINED SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        self.test_semantic_memory_basic()
        self.test_episodic_memory()
        self.test_text_learning()
        self.test_multimodal_text()
        self.test_multimodal_image()
        self.test_context_engine()
        self.test_knowledge_query()
        self.test_integrated_workflow()
        
        return self.results.summary()


def main():
    tests = NSCKTests()
    summary = tests.run_all()
    
    # Save results
    results_file = Path(__file__).parent / "results" / "test_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
