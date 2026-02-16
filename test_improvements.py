#!/usr/bin/env python3
"""
Test Improvements with Real-World Data
=======================================

This script tests the enhanced NSCK AI model with real-world data after
implementing the improvements identified in comprehensive testing:

1. Enhanced response generation (ResponseComposer)
2. Improved context retention (ContextRetentionModule)  
3. Real-world training data (RealWorldDataLoader)

Usage:
    python test_improvements.py [--verbose]
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.response_composer import ResponseComposer
from nsck_ai_model.context_retention import ContextRetentionModule
from nsck_ai_model.realworld_data_loader import RealWorldDataLoader
from nsck_ai_model.comprehensive_benchmark import ComprehensiveBenchmark
from nsck_ai_model.telemetry_monitor import TelemetryMonitor, MonitoredEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
)
logger = logging.getLogger("test_improvements")


class EnhancedNSCKAIEngine(NSCKAIEngine):
    """
    Enhanced version of NSCK AI Engine with improvements.
    """
    
    def __init__(self):
        """Initialize enhanced engine."""
        super().__init__()
        
        # Add enhancement modules
        self.response_composer = ResponseComposer()
        self.context_module = ContextRetentionModule(max_history=20, attention_window=5)
        
        logger.info("Enhanced NSCK AI Engine initialized with improvements")
    
    def train_on_text(self, text: str) -> Dict[str, Any]:
        """
        Train on text and also learn response patterns.
        
        Args:
            text: Text to train on
            
        Returns:
            Training result
        """
        # Standard training
        result = super().train_on_text(text)
        
        # Also learn patterns for response composer
        self.response_composer.learn_from_text(text)
        
        return result
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced chat with context retention and better response generation.
        
        Args:
            user_input: User's input
            
        Returns:
            Response dictionary
        """
        # Check if we should use context
        use_context = self.context_module.should_use_context(user_input)
        
        # Get enhanced query with context
        if use_context:
            enhanced_input, enhanced_concepts = self.context_module.enhance_query_with_context(
                user_input,
                self._extract_concepts(user_input.lower())
            )
        else:
            enhanced_input = user_input
            enhanced_concepts = self._extract_concepts(user_input.lower())
        
        # Get base response from parent
        result = super().chat(enhanced_input)
        
        # Extract query concepts
        query_concepts = self._extract_concepts(user_input.lower())
        
        # Get retrieved information
        retrieved_sentences = []
        if result.get('response'):
            # Parse the response to get source sentences
            retrieved_sentences = [result['response']]
        
        # Try to get better response using composer
        try:
            # Get activation and facts from trace if available
            activation = {}
            related_facts = []
            
            if 'trace' in result and 'steps' in result['trace']:
                for step in result['trace']['steps']:
                    if step.get('stage') == 'spread_activation':
                        activation = step.get('outputs', {}).get('activated', {})
                    elif step.get('stage') == 'query_knowledge':
                        related_facts = step.get('outputs', {}).get('facts', [])
            
            # Compose enhanced response
            composed_response = self.response_composer.compose_response(
                query=user_input,
                query_concepts=query_concepts,
                retrieved_sentences=retrieved_sentences,
                activation=activation,
                related_facts=related_facts,
                confidence=result.get('confidence', 0.5)
            )
            
            # Use composed response if it's better (longer and confident)
            if composed_response and len(composed_response) >= len(result['response']) * 0.8:
                result['response'] = composed_response
                result['enhanced'] = True
            
        except Exception as e:
            logger.warning(f"Response composition failed: {e}")
            result['enhanced'] = False
        
        # Add to context history
        self.context_module.add_turn(
            user_input=user_input,
            response=result['response'],
            concepts=query_concepts
        )
        
        return result
    
    def get_context_summary(self) -> str:
        """Get conversation context summary."""
        return self.context_module.get_context_summary()
    
    def clear_context(self):
        """Clear conversation context."""
        self.context_module.clear_context()


def test_context_retention_improvement():
    """Test context retention with enhanced module."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: CONTEXT RETENTION IMPROVEMENT")
    logger.info("=" * 70)
    
    engine = EnhancedNSCKAIEngine()
    
    # Train on context-building data
    training = [
        "Alice is a scientist who works at MIT.",
        "Alice studies artificial intelligence and cognitive science.",
        "MIT is located in Cambridge, Massachusetts.",
        "Artificial intelligence is a field of computer science.",
    ]
    
    for text in training:
        engine.train_on_text(text)
    
    # Test context retention
    conversation = [
        "Who is Alice?",
        "Where does she work?",
        "What does she study?",
        "Where is that university?",
    ]
    
    results = []
    correct = 0
    
    for query in conversation:
        result = engine.chat(query)
        results.append({
            'query': query,
            'response': result['response'],
            'confidence': result['confidence']
        })
        
        # Check if answer is correct
        query_lower = query.lower()
        response_lower = result['response'].lower()
        
        if 'who' in query_lower and 'alice' in response_lower:
            correct += 1
        elif 'where does she' in query_lower and 'mit' in response_lower:
            correct += 1
        elif 'what does she' in query_lower and ('artificial' in response_lower or 'cognitive' in response_lower):
            correct += 1
        elif 'where is that' in query_lower and ('cambridge' in response_lower or 'massachusetts' in response_lower):
            correct += 1
    
    score = correct / len(conversation) if conversation else 0
    
    print(f"\n  Context Retention Score: {score:.1%} ({correct}/{len(conversation)} correct)")
    print(f"  Previous Score: 25%")
    print(f"  Improvement: {(score - 0.25) / 0.25 * 100:+.0f}%")
    
    for i, r in enumerate(results):
        print(f"\n  Q{i+1}: {r['query']}")
        print(f"  A{i+1}: {r['response']}")
        print(f"  Confidence: {r['confidence']:.2f}")
    
    return score


def test_response_quality_improvement():
    """Test response generation quality with enhanced composer."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: RESPONSE QUALITY IMPROVEMENT")
    logger.info("=" * 70)
    
    engine = EnhancedNSCKAIEngine()
    
    # Train on sample data
    training = [
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Plants use chlorophyll to absorb light energy.",
        "The process produces glucose and oxygen as byproducts.",
        "Water and carbon dioxide are required for photosynthesis.",
    ]
    
    for text in training:
        engine.train_on_text(text)
    
    # Test response quality
    queries = [
        "What is photosynthesis?",
        "How do plants use sunlight?",
        "What does photosynthesis produce?",
    ]
    
    responses = []
    
    for query in queries:
        result = engine.chat(query)
        responses.append({
            'query': query,
            'response': result['response'],
            'confidence': result['confidence'],
            'enhanced': result.get('enhanced', False),
            'length': len(result['response'])
        })
    
    # Calculate metrics
    avg_length = sum(r['length'] for r in responses) / len(responses)
    avg_confidence = sum(r['confidence'] for r in responses) / len(responses)
    enhanced_count = sum(1 for r in responses if r.get('enhanced'))
    
    print(f"\n  Average Response Length: {avg_length:.0f} characters")
    print(f"  Average Confidence: {avg_confidence:.2%}")
    print(f"  Enhanced Responses: {enhanced_count}/{len(responses)}")
    
    for i, r in enumerate(responses):
        print(f"\n  Q{i+1}: {r['query']}")
        print(f"  A{i+1}: {r['response']}")
        print(f"  Enhanced: {'Yes' if r.get('enhanced') else 'No'}")
    
    return avg_confidence


def test_with_realworld_data():
    """Test with real-world data."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: REAL-WORLD DATA TRAINING & TESTING")
    logger.info("=" * 70)
    
    # Load real-world data
    data_loader = RealWorldDataLoader()
    all_data = data_loader.load_all_data()
    
    # Create engine
    engine = EnhancedNSCKAIEngine()
    
    # Train on real-world data
    training_start = time.time()
    trained_samples = 0
    
    for category, texts in all_data.items():
        logger.info(f"\n  Training on {category} data ({len(texts)} samples)...")
        for text in texts:
            engine.train_on_text(text)
            trained_samples += 1
    
    training_time = time.time() - training_start
    
    # Get stats
    stats = engine.get_system_stats()
    
    print(f"\n  Trained on {trained_samples} real-world samples")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Concepts learned: {stats['knowledge']['total_concepts']}")
    print(f"  Relations learned: {stats['knowledge']['total_relations']}")
    print(f"  Episodes stored: {stats['knowledge']['total_episodes']}")
    
    # Test with diverse queries
    test_queries = [
        "What is photosynthesis?",
        "Tell me about the Roman Empire",
        "How does machine learning work?",
        "What is the water cycle?",
        "Explain democracy",
        "What is DNA?",
        "Tell me about climate change",
        "What is a REST API?",
    ]
    
    print(f"\n  Testing with {len(test_queries)} queries...")
    
    correct_responses = 0
    total_confidence = 0
    
    for query in test_queries:
        result = engine.chat(query)
        total_confidence += result['confidence']
        
        # Check if response is reasonable (contains relevant keywords)
        response_lower = result['response'].lower()
        query_lower = query.lower()
        
        # Simple relevance check
        if any(word in response_lower for word in query_lower.split() if len(word) > 3):
            correct_responses += 1
        
        print(f"\n  Q: {query}")
        print(f"  A: {result['response'][:100]}...")
        print(f"  Confidence: {result['confidence']:.2f}")
    
    success_rate = correct_responses / len(test_queries)
    avg_confidence = total_confidence / len(test_queries)
    
    print(f"\n  Success Rate: {success_rate:.1%} ({correct_responses}/{len(test_queries)})")
    print(f"  Average Confidence: {avg_confidence:.2%}")
    
    return {
        'samples_trained': trained_samples,
        'training_time': training_time,
        'success_rate': success_rate,
        'avg_confidence': avg_confidence,
        'concepts': stats['knowledge']['total_concepts'],
    }


def run_comprehensive_comparison():
    """Run comprehensive benchmark and compare with previous results."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: COMPREHENSIVE BENCHMARK COMPARISON")
    logger.info("=" * 70)
    
    # Load previous results
    previous_results = {
        'cognitive_pass_rate': 0.667,
        'context_retention': 0.25,
        'avg_confidence': 0.6391,
        'avg_latency_ms': 3.43,
    }
    
    # Create benchmark with enhanced engine
    benchmark = ComprehensiveBenchmark(output_dir="./improved_benchmark_results")
    
    # Override engine creation to use enhanced version
    benchmark.engine = EnhancedNSCKAIEngine()
    benchmark.pipeline.engine = benchmark.engine
    
    # Run training
    benchmark.run_comprehensive_training()
    
    # Run cognitive tests
    cognitive_results = benchmark.test_cognitive_capabilities()
    
    # Get key metrics
    new_results = {
        'cognitive_pass_rate': cognitive_results['pass_rate'],
        'avg_score': cognitive_results['average_score'],
    }
    
    # Find context retention test result
    for test in cognitive_results['results']:
        if test['test_name'] == 'Context Retention':
            new_results['context_retention'] = test['score']
    
    # Compare
    print("\n  COMPARISON WITH PREVIOUS RESULTS:")
    print("  " + "-" * 60)
    
    for metric, old_value in previous_results.items():
        new_value = new_results.get(metric)
        if new_value is not None:
            improvement = (new_value - old_value) / old_value * 100 if old_value > 0 else 0
            status = "✓ IMPROVED" if improvement > 0 else "✗ DECLINED"
            print(f"  {metric:20s}: {old_value:.2%} → {new_value:.2%} ({improvement:+.0f}%) {status}")
    
    return new_results


def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("TESTING IMPROVEMENTS WITH REAL-WORLD DATA")
    print("=" * 80)
    
    results = {}
    
    try:
        # Test 1: Context retention
        results['context_retention'] = test_context_retention_improvement()
        
        # Test 2: Response quality
        results['response_quality'] = test_response_quality_improvement()
        
        # Test 3: Real-world data
        results['realworld'] = test_with_realworld_data()
        
        # Test 4: Comprehensive comparison
        results['comparison'] = run_comprehensive_comparison()
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY OF IMPROVEMENTS")
        print("=" * 80)
        
        print(f"\n  Context Retention: {results['context_retention']:.1%} (was 25%)")
        print(f"  Response Quality: {results['response_quality']:.1%} avg confidence")
        print(f"  Real-World Success: {results['realworld']['success_rate']:.1%}")
        print(f"  Samples Trained: {results['realworld']['samples_trained']}")
        print(f"  Concepts Learned: {results['realworld']['concepts']}")
        
        # Save results
        with open('improvement_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n  Results saved to: improvement_test_results.json")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Testing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
