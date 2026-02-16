"""
Extended Training Pipeline - Train with expanded web data and test rigorously
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Import our components
from web_data_collector import WebDataCollector
from neural_chat_backend import TrainableChatBackend
from improved_backend import ImprovedBackend
from comprehensive_test_suite import ComprehensiveTestSuite


class ExtendedTrainingPipeline:
    """Pipeline for training with web data and comprehensive testing"""
    
    def __init__(self, output_dir: str = "extended_training_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.collector = WebDataCollector()
        self.results = {}
        
    def step1_load_web_data(self) -> List[str]:
        """Load collected web data"""
        print("\n" + "="*80)
        print("STEP 1: LOADING WEB DATA")
        print("="*80)
        
        training_data = self.collector.get_training_data()
        
        if not training_data:
            print("No cached data found. Collecting now...")
            self.collector.collect_all(target_count=500)
            training_data = self.collector.get_training_data()
            
        print(f"\n✓ Loaded {len(training_data)} sentences for training")
        
        # Save to session directory
        data_file = self.session_dir / "training_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
            
        self.results['training_data_count'] = len(training_data)
        return training_data
        
    def step2_train_base_model(self, training_data: List[str]) -> TrainableChatBackend:
        """Train standard backend"""
        print("\n" + "="*80)
        print("STEP 2: TRAINING BASE MODEL")
        print("="*80)
        
        print("\nInitializing TrainableChatBackend...")
        backend = TrainableChatBackend()
        
        print(f"Training on {len(training_data)} sentences...")
        
        # Track training progress
        batch_size = 50
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            for sentence in batch:
                try:
                    backend.learn_from_text(sentence)
                except Exception as e:
                    print(f"Warning: Failed to learn '{sentence[:50]}...': {e}")
                    
            progress = min(i + batch_size, len(training_data))
            print(f"  Progress: {progress}/{len(training_data)} ({100*progress//len(training_data)}%)")
            
        # Get stats from semantic memory
        print(f"\n✓ Training complete!")
        
        try:
            # Try to get concept count from semantic memory
            if hasattr(backend.semantic, 'concepts'):
                concepts_count = len(backend.semantic.concepts)
            elif hasattr(backend.semantic, '_concepts'):
                concepts_count = len(backend.semantic._concepts)
            else:
                concepts_count = backend.training_metadata.get('texts_learned', 0)
                
            # Try to get facts count
            if hasattr(backend.text_learner, 'learned_facts'):
                facts_count = len(backend.text_learner.learned_facts)
            else:
                facts_count = backend.training_metadata.get('total_facts', 0)
                
            print(f"  Concepts learned: {concepts_count}")
            print(f"  Facts stored: {facts_count}")
            
            self.results['base_model_concepts'] = concepts_count
            self.results['base_model_facts'] = facts_count
        except Exception as e:
            print(f"  Could not retrieve stats: {e}")
            self.results['base_model_concepts'] = 0
            self.results['base_model_facts'] = 0
        
        # Save model
        model_file = self.session_dir / "base_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(backend, f)
        print(f"  Saved to: {model_file}")
        
        return backend
        
    def step3_create_improved_model(self, base_backend: TrainableChatBackend) -> ImprovedBackend:
        """Wrap base model with improvements"""
        print("\n" + "="*80)
        print("STEP 3: CREATING IMPROVED MODEL")
        print("="*80)
        
        print("\nWrapping with ImprovedBackend (adaptive retrieval + episodic memory + multi-hop)...")
        improved = ImprovedBackend(base_backend)
        
        # Save improved model
        model_file = self.session_dir / "improved_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(improved, f)
        print(f"✓ Saved to: {model_file}")
        
        return improved
        
    def step4_test_base_model(self, backend: TrainableChatBackend) -> Dict[str, Any]:
        """Run comprehensive tests on base model"""
        print("\n" + "="*80)
        print("STEP 4: TESTING BASE MODEL")
        print("="*80)
        
        test_suite = ComprehensiveTestSuite(backend)
        results = test_suite.run_all_tests()
        
        if results is None:
            print("✗ Base model testing failed - model could not be loaded")
            return {'pass_rate': 0.0, 'passed': 0, 'total_tests': 0, 'error': 'Model load failed'}
            
        print(f"\nBase Model Results:")
        print(f"  Pass Rate: {results['pass_rate']:.1f}%")
        print(f"  Tests: {results['passed']}/{results['total_tests']}")
        
        self.results['base_pass_rate'] = results['pass_rate']
        self.results['base_passed'] = results['passed']
        self.results['base_total'] = results['total_tests']
        
        return results
        
    def step5_test_improved_model(self, backend: ImprovedBackend) -> Dict[str, Any]:
        """Run comprehensive tests on improved model"""
        print("\n" + "="*80)
        print("STEP 5: TESTING IMPROVED MODEL")
        print("="*80)
        
        test_suite = ComprehensiveTestSuite(backend)
        results = test_suite.run_all_tests()
        
        if results is None:
            print("✗ Improved model testing failed - model could not be loaded")
            return {'pass_rate': 0.0, 'passed': 0, 'total_tests': 0, 'error': 'Model load failed'}
            
        print(f"\nImproved Model Results:")
        print(f"  Pass Rate: {results['pass_rate']:.1f}%")
        print(f"  Tests: {results['passed']}/{results['total_tests']}")
        
        self.results['improved_pass_rate'] = results['pass_rate']
        self.results['improved_passed'] = results['passed']
        self.results['improved_total'] = results['total_tests']
        
        return results
        
    def step6_run_custom_scenarios(self, backend: ImprovedBackend) -> Dict[str, Any]:
        """Run custom real-world conversation scenarios"""
        print("\n" + "="*80)
        print("STEP 6: CUSTOM REAL-WORLD SCENARIOS")
        print("="*80)
        
        scenarios = [
            {
                'name': 'Science Knowledge Chain',
                'turns': [
                    "What is the speed of light?",
                    "How does that relate to Einstein's theory?",
                    "Why is this important for space travel?"
                ]
            },
            {
                'name': 'Historical Context',
                'turns': [
                    "When was World War II?",
                    "What were the main causes?",
                    "How did it change the world?"
                ]
            },
            {
                'name': 'Technology Evolution',
                'turns': [
                    "When was the Internet invented?",
                    "How has it changed communication?",
                    "What are the future implications?"
                ]
            },
            {
                'name': 'Biology Connections',
                'turns': [
                    "What is photosynthesis?",
                    "Why is it important for life?",
                    "What happens without it?"
                ]
            },
            {
                'name': 'Cross-Domain Reasoning',
                'turns': [
                    "Tell me about quantum mechanics",
                    "How does this relate to computing?",
                ]
            }
        ]
        
        passed = 0
        total = 0
        
        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            scenario_passed = True
            
            for i, query in enumerate(scenario['turns'], 1):
                print(f"  Q{i}: {query}")
                try:
                    response = backend.query(query)
                    
                    # Check if response is substantive
                    if len(response) < 50:
                        print(f"     ✗ Response too short: {len(response)} chars")
                        scenario_passed = False
                    elif "don't know" in response.lower() or "cannot" in response.lower():
                        print(f"     ✗ Non-committal response")
                        scenario_passed = False
                    else:
                        print(f"     ✓ Good response: {len(response)} chars")
                        
                    total += 1
                    
                except Exception as e:
                    print(f"     ✗ Error: {e}")
                    scenario_passed = False
                    total += 1
                    
            if scenario_passed:
                print(f"  → Scenario PASSED")
                passed += 1
            else:
                print(f"  → Scenario FAILED")
                
        pass_rate = 100 * passed / len(scenarios) if scenarios else 0
        print(f"\nCustom Scenarios: {passed}/{len(scenarios)} passed ({pass_rate:.1f}%)")
        
        self.results['custom_scenarios_passed'] = passed
        self.results['custom_scenarios_total'] = len(scenarios)
        self.results['custom_scenarios_pass_rate'] = pass_rate
        
        return {
            'passed': passed,
            'total': len(scenarios),
            'pass_rate': pass_rate
        }
        
    def step7_generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("STEP 7: GENERATING REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("EXTENDED TRAINING RESULTS")
        report.append("="*80)
        report.append(f"Session: {self.session_id}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("TRAINING DATA:")
        report.append(f"  Sentences: {self.results.get('training_data_count', 0)}")
        report.append(f"  Concepts learned: {self.results.get('base_model_concepts', 0)}")
        report.append(f"  Facts stored: {self.results.get('base_model_facts', 0)}")
        report.append("")
        
        report.append("BASE MODEL PERFORMANCE:")
        report.append(f"  Pass Rate: {self.results.get('base_pass_rate', 0):.1f}%")
        report.append(f"  Tests: {self.results.get('base_passed', 0)}/{self.results.get('base_total', 0)}")
        report.append("")
        
        report.append("IMPROVED MODEL PERFORMANCE:")
        report.append(f"  Comprehensive Tests: {self.results.get('improved_pass_rate', 0):.1f}%")
        report.append(f"  Tests: {self.results.get('improved_passed', 0)}/{self.results.get('improved_total', 0)}")
        report.append(f"  Custom Scenarios: {self.results.get('custom_scenarios_pass_rate', 0):.1f}%")
        report.append(f"  Scenarios: {self.results.get('custom_scenarios_passed', 0)}/{self.results.get('custom_scenarios_total', 0)}")
        report.append("")
        
        # Calculate improvement
        base_rate = self.results.get('base_pass_rate', 0)
        improved_rate = self.results.get('improved_pass_rate', 0)
        improvement = improved_rate - base_rate
        
        report.append("IMPROVEMENT ANALYSIS:")
        report.append(f"  Base → Improved: {base_rate:.1f}% → {improved_rate:.1f}%")
        if improvement > 0:
            report.append(f"  ✅ IMPROVEMENT: +{improvement:.1f} percentage points")
        elif improvement < 0:
            report.append(f"  ⚠️ REGRESSION: {improvement:.1f} percentage points")
        else:
            report.append(f"  → No change")
        report.append("")
        
        # Issues found
        report.append("ISSUES IDENTIFIED:")
        if base_rate < 70:
            report.append("  ⚠️ Base model performance below 70% - needs more training data")
        if improved_rate < 80:
            report.append("  ⚠️ Improved model below 80% - retrieval/reasoning needs tuning")
        if improvement < 10:
            report.append("  ⚠️ Improvement less than 10% - fixes may not be effective enough")
        if not any(['⚠️' in line for line in report[-3:]]):
            report.append("  ✓ No major issues detected")
        report.append("")
        
        report.append("="*80)
        
        # Print report
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # Save report
        report_file = self.session_dir / "report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        # Save JSON results
        json_file = self.session_dir / "results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n✓ Reports saved to: {self.session_dir}")
        
    def run_full_pipeline(self):
        """Execute complete training and testing pipeline"""
        print("\n" + "="*80)
        print("EXTENDED TRAINING PIPELINE")
        print("="*80)
        print("This will:")
        print("  1. Load web-collected training data")
        print("  2. Train base model")
        print("  3. Create improved model with all fixes")
        print("  4. Test base model comprehensively")
        print("  5. Test improved model comprehensively")
        print("  6. Run custom real-world scenarios")
        print("  7. Generate comparison report")
        print("="*80)
        
        try:
            # Step 1: Load data
            training_data = self.step1_load_web_data()
            
            # Step 2: Train base
            base_backend = self.step2_train_base_model(training_data)
            
            # Step 3: Create improved
            improved_backend = self.step3_create_improved_model(base_backend)
            
            # Step 4: Test base
            base_results = self.step4_test_base_model(base_backend)
            
            # Step 5: Test improved
            improved_results = self.step5_test_improved_model(improved_backend)
            
            # Step 6: Custom scenarios
            scenario_results = self.step6_run_custom_scenarios(improved_backend)
            
            # Step 7: Report
            self.step7_generate_report()
            
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE")
            print("="*80)
            
            return self.results
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Run the extended training pipeline"""
    pipeline = ExtendedTrainingPipeline()
    results = pipeline.run_full_pipeline()
    
    if results:
        print("\n✓ Training and testing completed successfully")
        return 0
    else:
        print("\n✗ Pipeline encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
