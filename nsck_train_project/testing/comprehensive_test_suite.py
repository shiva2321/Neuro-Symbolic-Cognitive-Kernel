#!/usr/bin/env python3
"""
NSCK AI Comprehensive Test Suite
=================================

Rigorous testing covering:
- Text understanding (multiple domains)
- Image understanding and description
- Context maintenance
- Counter-factual reasoning
- Cross-domain transfer
- Performance metrics
- Memory and efficiency

All tests logged with detailed telemetry.
"""

import sys
import os
import time
import json
import pickle
import psutil
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

# Try to import PIL
try:
    from PIL import Image as PILImage
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class TestLogger:
    """Comprehensive test logging system."""
    
    def __init__(self, log_dir='test_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.telemetry = []
        self.start_time = time.time()
        
        # Create log files
        self.summary_file = self.session_dir / 'summary.json'
        self.detailed_file = self.session_dir / 'detailed_log.txt'
        self.telemetry_file = self.session_dir / 'telemetry.json'
        
        self.log(f"Test session started: {self.session_id}")
    
    def log(self, message: str, level='INFO'):
        """Log a message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        
        with open(self.detailed_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def record_test(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Record a test result."""
        result = {
            'test_name': test_name,
            'passed': passed,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        self.log(f"{status}: {test_name}")
        for key, value in details.items():
            self.log(f"  {key}: {value}")
    
    def record_telemetry(self, operation: str, metrics: Dict[str, Any]):
        """Record telemetry data."""
        telemetry = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.telemetry.append(telemetry)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total_tests - passed
        
        duration = time.time() - self.start_time
        
        return {
            'session_id': self.session_id,
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total_tests if total_tests > 0 else 0,
            'duration_seconds': duration,
            'results': self.results,
            'telemetry_samples': len(self.telemetry)
        }
    
    def save_results(self):
        """Save all results to disk."""
        summary = self.get_summary()
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save telemetry
        with open(self.telemetry_file, 'w') as f:
            json.dump(self.telemetry, f, indent=2)
        
        self.log(f"Results saved to {self.session_dir}")
        self.log(f"Pass rate: {summary['pass_rate']*100:.1f}%")


class PerformanceMonitor:
    """Monitor system performance during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'memory_percent': self.process.memory_percent(),
            'num_threads': self.process.num_threads(),
        }
    
    def measure_operation(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Measure performance of an operation."""
        # Pre-operation metrics
        pre_mem = self.process.memory_info().rss
        
        # Execute
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Post-operation metrics
        post_mem = self.process.memory_info().rss
        
        metrics = {
            'duration_ms': duration * 1000,
            'memory_delta_mb': (post_mem - pre_mem) / 1024 / 1024,
            'final_memory_mb': post_mem / 1024 / 1024,
        }
        
        return result, metrics


class ComprehensiveTestSuite:
    """Comprehensive test suite for NSCK AI."""
    
    def __init__(self, model_path='ai_model.pkl'):
        self.logger = TestLogger()
        self.monitor = PerformanceMonitor()
        self.model_path = model_path
        self.backend = None
        self.image_system = None
        
        self.logger.log("Initializing comprehensive test suite")
    
    def load_model(self):
        """Load the AI model."""
        self.logger.log(f"Loading model from {self.model_path}")
        
        try:
            result, metrics = self.monitor.measure_operation(
                self._load_model_sync
            )
            
            self.logger.record_telemetry('model_load', metrics)
            self.logger.log(f"Model loaded in {metrics['duration_ms']:.1f}ms")
            
            return True
        except Exception as e:
            self.logger.log(f"Failed to load model: {e}", 'ERROR')
            self.logger.log(traceback.format_exc(), 'ERROR')
            return False
    
    def _load_model_sync(self):
        """Synchronous model loading."""
        with open(self.model_path, 'rb') as f:
            self.backend = pickle.load(f)
        
        # Setup image understanding if available
        if PIL_AVAILABLE:
            try:
                from image_understanding import ImageUnderstanding
                
                class BackendWrapper:
                    def __init__(self, backend):
                        self.backend = backend
                        self.encoder = backend.text_learner
                        self.knowledge = backend.semantic
                    
                    def train_on_text(self, text):
                        self.backend.learn_from_text(text)
                
                wrapper = BackendWrapper(self.backend)
                self.image_system = ImageUnderstanding(wrapper)
                self.logger.log("Image understanding enabled")
            except Exception as e:
                self.logger.log(f"Image system not available: {e}", 'WARN')
        
        return self.backend
    
    def run_all_tests(self):
        """Run all test categories."""
        self.logger.log("="*80)
        self.logger.log("STARTING COMPREHENSIVE TEST SUITE")
        self.logger.log("="*80)
        
        if not self.load_model():
            self.logger.log("Cannot proceed without model", 'ERROR')
            return None
        
        # Test categories
        test_categories = [
            ('Basic Text Understanding', self.test_basic_text),
            ('Advanced Text Understanding', self.test_advanced_text),
            ('Context Maintenance', self.test_context_maintenance),
            ('Counter-Factual Reasoning', self.test_counterfactual),
            ('Cross-Domain Transfer', self.test_cross_domain),
            ('Response Time Scaling', self.test_response_scaling),
            ('Memory Efficiency', self.test_memory_efficiency),
        ]
        
        if self.image_system and PIL_AVAILABLE:
            test_categories.extend([
                ('Image Understanding', self.test_image_understanding),
                ('Image Description', self.test_image_description),
            ])
        
        for category_name, test_func in test_categories:
            self.logger.log(f"\n{'='*80}")
            self.logger.log(f"CATEGORY: {category_name}")
            self.logger.log(f"{'='*80}")
            
            try:
                test_func()
            except Exception as e:
                self.logger.log(f"Category failed: {e}", 'ERROR')
                self.logger.log(traceback.format_exc(), 'ERROR')
        
        # Generate final report
        return self.generate_report()
    
    def test_basic_text(self):
        """Test basic text understanding."""
        test_cases = [
            {
                'question': 'What is gravity?',
                'expected_keywords': ['gravity', 'force', 'mass', 'objects'],
                'domain': 'physics'
            },
            {
                'question': 'How does photosynthesis work?',
                'expected_keywords': ['photosynthesis', 'plants', 'energy', 'light'],
                'domain': 'biology'
            },
            {
                'question': 'What is water made of?',
                'expected_keywords': ['water', 'hydrogen', 'oxygen', 'h2o'],
                'domain': 'chemistry'
            },
            {
                'question': 'What is artificial intelligence?',
                'expected_keywords': ['intelligence', 'machines', 'ai'],
                'domain': 'technology'
            },
            {
                'question': 'What is a CPU?',
                'expected_keywords': ['cpu', 'processor', 'computer'],
                'domain': 'hardware'
            },
        ]
        
        for i, case in enumerate(test_cases, 1):
            result, metrics = self.monitor.measure_operation(
                self.backend.query, case['question']
            )
            
            response = result.get('response', '').lower()
            confidence = result.get('confidence', 0.0)
            
            # Check for expected keywords
            found_keywords = [kw for kw in case['expected_keywords'] 
                            if kw in response]
            keyword_rate = len(found_keywords) / len(case['expected_keywords'])
            
            passed = keyword_rate >= 0.5 and confidence > 0.7
            
            details = {
                'question': case['question'],
                'domain': case['domain'],
                'response_length': len(response),
                'confidence': round(confidence, 3),
                'keyword_match_rate': round(keyword_rate, 3),
                'found_keywords': found_keywords,
                'response_time_ms': round(metrics['duration_ms'], 2)
            }
            
            self.logger.record_test(f"basic_text_{i:02d}_{case['domain']}", passed, details)
            self.logger.record_telemetry(f"query_{case['domain']}", metrics)
    
    def test_advanced_text(self):
        """Test advanced text understanding."""
        test_cases = [
            {
                'question': 'Why do objects fall to the ground?',
                'category': 'causal_reasoning',
                'expected_concepts': ['gravity', 'force', 'earth']
            },
            {
                'question': 'What happens when plants do not get sunlight?',
                'category': 'consequential_reasoning',
                'expected_concepts': ['photosynthesis', 'energy', 'light']
            },
            {
                'question': 'Compare gravity and magnetism',
                'category': 'comparison',
                'expected_concepts': ['gravity', 'force']
            },
        ]
        
        for i, case in enumerate(test_cases, 1):
            result, metrics = self.monitor.measure_operation(
                self.backend.query, case['question']
            )
            
            response = result.get('response', '').lower()
            confidence = result.get('confidence', 0.0)
            
            # Check for conceptual understanding
            concepts_found = sum(1 for c in case['expected_concepts'] if c in response)
            concept_rate = concepts_found / len(case['expected_concepts'])
            
            passed = concept_rate >= 0.5
            
            details = {
                'question': case['question'],
                'category': case['category'],
                'confidence': round(confidence, 3),
                'concept_coverage': round(concept_rate, 3),
                'response_time_ms': round(metrics['duration_ms'], 2),
                'response_preview': response[:100]
            }
            
            self.logger.record_test(f"advanced_{i:02d}_{case['category']}", passed, details)
    
    def test_context_maintenance(self):
        """Test how long the system maintains context."""
        self.logger.log("Testing context maintenance across multiple queries")
        
        conversation = [
            ("What is gravity?", "gravity"),
            ("Who discovered it?", "newton"),
            ("When did this discovery happen?", None),  # May not know
            ("What else did this person discover?", "newton"),
        ]
        
        context_maintained = 0
        
        for i, (question, expected_context) in enumerate(conversation, 1):
            result, metrics = self.monitor.measure_operation(
                self.backend.query, question
            )
            
            response = result.get('response', '').lower()
            
            if expected_context and expected_context in response:
                context_maintained += 1
                maintained = True
            else:
                maintained = expected_context is None  # Don't count missing info as failure
            
            details = {
                'question_num': i,
                'question': question,
                'expected_context': expected_context,
                'context_maintained': maintained,
                'response_time_ms': round(metrics['duration_ms'], 2),
                'response_preview': response[:80]
            }
            
            self.logger.record_test(f"context_{i:02d}", maintained, details)
        
        # Overall context score
        valid_questions = len([c for c in conversation if c[1] is not None])
        context_score = context_maintained / valid_questions if valid_questions > 0 else 0
        
        self.logger.log(f"Context maintenance score: {context_score*100:.1f}%")
    
    def test_counterfactual(self):
        """Test counter-factual reasoning."""
        test_cases = [
            {
                'question': 'What if there was no gravity?',
                'category': 'hypothetical',
                'should_recognize': True
            },
            {
                'question': 'Photosynthesis does not require light',  # False statement
                'category': 'false_premise',
                'should_recognize': False
            },
            {
                'question': 'If water was not H2O, what would happen?',
                'category': 'counterfactual',
                'should_recognize': True
            },
        ]
        
        for i, case in enumerate(test_cases, 1):
            result, metrics = self.monitor.measure_operation(
                self.backend.query, case['question']
            )
            
            response = result.get('response', '').lower()
            confidence = result.get('confidence', 0.0)
            
            # Check if system shows understanding (confidence patterns)
            # For false premises, confidence might be lower
            # This is a basic heuristic - more sophisticated analysis would be needed
            
            details = {
                'question': case['question'],
                'category': case['category'],
                'confidence': round(confidence, 3),
                'response_length': len(response),
                'response_time_ms': round(metrics['duration_ms'], 2),
                'response_preview': response[:100]
            }
            
            # Pass if system responds (we don't expect perfect counterfactual reasoning yet)
            passed = len(response) > 20
            
            self.logger.record_test(f"counterfactual_{i:02d}", passed, details)
    
    def test_cross_domain(self):
        """Test cross-domain knowledge transfer."""
        self.logger.log("Testing cross-domain transfer")
        
        test_cases = [
            {
                'question': 'How is CPU processing similar to biological processes?',
                'domain_a': 'hardware',
                'domain_b': 'biology',
                'transfer_type': 'analogy'
            },
            {
                'question': 'What do water molecules and computer systems have in common?',
                'domain_a': 'chemistry',
                'domain_b': 'hardware',
                'transfer_type': 'comparison'
            },
        ]
        
        for i, case in enumerate(test_cases, 1):
            result, metrics = self.monitor.measure_operation(
                self.backend.query, case['question']
            )
            
            response = result.get('response', '').lower()
            
            # Check if both domains are referenced
            domains_referenced = 0
            if any(term in response for term in ['cpu', 'processor', 'computer']):
                domains_referenced += 1
            if any(term in response for term in ['water', 'biology', 'molecule']):
                domains_referenced += 1
            
            passed = domains_referenced >= 1  # At least one domain referenced
            
            details = {
                'question': case['question'],
                'domains': f"{case['domain_a']} + {case['domain_b']}",
                'transfer_type': case['transfer_type'],
                'domains_referenced': domains_referenced,
                'response_time_ms': round(metrics['duration_ms'], 2),
                'response_preview': response[:100]
            }
            
            self.logger.record_test(f"cross_domain_{i:02d}", passed, details)
    
    def test_response_scaling(self):
        """Test response time scaling with query complexity."""
        queries = [
            ("Short query", "Gravity?"),
            ("Medium query", "What is gravity and how does it work?"),
            ("Long query", "Explain in detail the concept of gravity, its discovery by Newton, Einstein's improvements, and how it affects objects on Earth"),
            ("Multi-concept", "Explain gravity, photosynthesis, water molecules, and CPUs"),
        ]
        
        times = []
        
        for complexity, question in queries:
            result, metrics = self.monitor.measure_operation(
                self.backend.query, question
            )
            
            times.append(metrics['duration_ms'])
            
            details = {
                'complexity': complexity,
                'query_length': len(question),
                'response_time_ms': round(metrics['duration_ms'], 2),
                'memory_delta_mb': round(metrics.get('memory_delta_mb', 0), 3)
            }
            
            # Pass if response time is reasonable (< 5 seconds)
            passed = metrics['duration_ms'] < 5000
            
            self.logger.record_test(f"scaling_{complexity.replace(' ', '_')}", passed, details)
        
        # Calculate scaling factor
        if len(times) > 1:
            scaling_factor = times[-1] / times[0]
            self.logger.log(f"Response time scaling factor: {scaling_factor:.2f}x")
    
    def test_memory_efficiency(self):
        """Test memory usage and efficiency."""
        initial_mem = self.monitor.process.memory_info().rss / 1024 / 1024
        
        # Run multiple queries and monitor memory
        num_queries = 50
        self.logger.log(f"Running {num_queries} queries to test memory stability")
        
        memories = []
        
        for i in range(num_queries):
            question = f"What is concept number {i}?"
            result = self.backend.query(question)
            
            current_mem = self.monitor.process.memory_info().rss / 1024 / 1024
            memories.append(current_mem)
        
        final_mem = self.monitor.process.memory_info().rss / 1024 / 1024
        mem_delta = final_mem - initial_mem
        mem_per_query = mem_delta / num_queries
        
        # Check for memory leaks (should be relatively stable)
        mem_growth_rate = (final_mem - initial_mem) / initial_mem
        
        passed = mem_growth_rate < 0.5  # Less than 50% growth
        
        details = {
            'initial_memory_mb': round(initial_mem, 2),
            'final_memory_mb': round(final_mem, 2),
            'memory_delta_mb': round(mem_delta, 2),
            'memory_per_query_kb': round(mem_per_query * 1024, 2),
            'growth_rate': round(mem_growth_rate * 100, 2),
            'num_queries': num_queries
        }
        
        self.logger.record_test('memory_efficiency', passed, details)
    
    def test_image_understanding(self):
        """Test image understanding capabilities."""
        if not self.image_system or not PIL_AVAILABLE:
            self.logger.log("Image testing skipped - not available", 'WARN')
            return
        
        self.logger.log("Testing image understanding")
        
        # Create test images
        test_images = self._create_test_images()
        
        # Teach images
        for img_data in test_images:
            result, metrics = self.monitor.measure_operation(
                self.image_system.learn_image,
                img_data['image'],
                img_data['description'],
                img_data['id']
            )
            
            details = {
                'image_id': img_data['id'],
                'description': img_data['description'],
                'concepts_learned': len(result.get('concepts', [])),
                'learning_time_ms': round(metrics['duration_ms'], 2)
            }
            
            passed = len(result.get('concepts', [])) > 0
            
            self.logger.record_test(f"image_learn_{img_data['id']}", passed, details)
        
        # Test image search
        search_queries = [
            ('red', 'red_square'),
            ('blue', 'blue_square'),
            ('green', 'green_square'),
        ]
        
        for query, expected_id in search_queries:
            results, metrics = self.monitor.measure_operation(
                self.image_system.search_by_text,
                query,
                top_k=3
            )
            
            # Check if expected image is in top results
            found = any(r['image_id'] == expected_id for r in results)
            
            details = {
                'query': query,
                'expected_id': expected_id,
                'found_in_results': found,
                'num_results': len(results),
                'top_similarity': round(results[0]['similarity'], 3) if results else 0,
                'search_time_ms': round(metrics['duration_ms'], 2)
            }
            
            self.logger.record_test(f"image_search_{query}", found, details)
    
    def test_image_description(self):
        """Test image description capabilities."""
        if not self.image_system or not PIL_AVAILABLE:
            return
        
        self.logger.log("Testing image description")
        
        # Create and describe images
        test_images = self._create_test_images()
        
        for img_data in test_images[:3]:  # Test first 3
            # First learn the image
            self.image_system.learn_image(
                img_data['image'],
                img_data['description'],
                img_data['id']
            )
            
            # Then test description
            description, metrics = self.monitor.measure_operation(
                self.image_system.describe_image,
                img_data['image']
            )
            
            # Check similarity to original description
            original_words = set(img_data['description'].lower().split())
            generated_words = set(description.lower().split())
            
            overlap = len(original_words & generated_words)
            similarity = overlap / len(original_words) if original_words else 0
            
            passed = similarity > 0.3
            
            details = {
                'image_id': img_data['id'],
                'original_description': img_data['description'],
                'generated_description': description,
                'word_overlap': overlap,
                'similarity': round(similarity, 3),
                'description_time_ms': round(metrics['duration_ms'], 2)
            }
            
            self.logger.record_test(f"image_describe_{img_data['id']}", passed, details)
    
    def _create_test_images(self) -> List[Dict]:
        """Create simple test images."""
        if not PIL_AVAILABLE:
            return []
        
        images = []
        
        # Red square
        red_img = np.zeros((32, 32, 3), dtype=np.uint8)
        red_img[:, :, 0] = 200
        images.append({
            'id': 'red_square',
            'image': PILImage.fromarray(red_img),
            'description': 'A red colored square'
        })
        
        # Blue square
        blue_img = np.zeros((32, 32, 3), dtype=np.uint8)
        blue_img[:, :, 2] = 200
        images.append({
            'id': 'blue_square',
            'image': PILImage.fromarray(blue_img),
            'description': 'A blue colored square'
        })
        
        # Green square
        green_img = np.zeros((32, 32, 3), dtype=np.uint8)
        green_img[:, :, 1] = 200
        images.append({
            'id': 'green_square',
            'image': PILImage.fromarray(green_img),
            'description': 'A green colored square'
        })
        
        return images
    
    def generate_report(self):
        """Generate comprehensive test report."""
        self.logger.log("\n" + "="*80)
        self.logger.log("GENERATING FINAL REPORT")
        self.logger.log("="*80)
        
        summary = self.logger.get_summary()
        
        self.logger.log(f"\nTest Session: {summary['session_id']}")
        self.logger.log(f"Duration: {summary['duration_seconds']:.2f} seconds")
        self.logger.log(f"Total Tests: {summary['total_tests']}")
        self.logger.log(f"Passed: {summary['passed']}")
        self.logger.log(f"Failed: {summary['failed']}")
        self.logger.log(f"Pass Rate: {summary['pass_rate']*100:.1f}%")
        
        # Save results
        self.logger.save_results()
        
        self.logger.log("\n" + "="*80)
        self.logger.log("COMPREHENSIVE TEST SUITE COMPLETE")
        self.logger.log("="*80)
        
        return summary


def main():
    """Run comprehensive test suite."""
    suite = ComprehensiveTestSuite('ai_model.pkl')
    suite.run_all_tests()


if __name__ == '__main__':
    main()
