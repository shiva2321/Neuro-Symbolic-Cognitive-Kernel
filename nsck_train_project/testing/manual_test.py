"""
Simple test of trained models - manual evaluation
"""

import pickle
import json
from pathlib import Path
from datetime import datetime


def test_model_manually(model, model_name="Model"):
    """
    Manual testing of a model with predefined questions    """
    
    print(f"\n{'='*80}")
    print(f"TESTING {model_name}")
    print(f"{'='*80}\n")
    
    test_questions = [
        # Science
        "What is the speed of light?",
        "What is photosynthesis?",
        "What is DNA?",
        "Tell me about gravity",
        "What is E=mc²?",
        
        # History
        "When was World War II?",
        "What was the Renaissance?",
        "When did humans land on the Moon?",
        "What was the Cold War?",
        "Who was Julius Caesar?",
        
        # Geography
        "What is Mount Everest?",
        "Where is the Sahara desert?",
        "What is the Amazon rainforest?",
        "Tell me about Antarctica",
        "What is the Pacific Ocean?",
        
        # Technology
        "When was the Internet invented?",
        "What is artificial intelligence?",
        "What is a computer?",
        "What is binary code?",
        "What is quantum computing?",
        
        # Math
        "What is Pi?",
        "What is the Pythagorean theorem?",
        "What are prime numbers?",
        "What is the golden ratio?",
        "What is calculus?",
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for i, question in enumerate(test_questions,1):
        print(f"[{i}/{len(test_questions)}] Q: {question}")
        
        try:
            result = model.query(question)
            
            # Handle both dict and string responses
            if isinstance(result, dict):
                response = result.get('response', str(result))
            else:
                response = str(result)
            
            # Evaluate response quality
            is_valid = (
                response and 
                len(response) >= 30 and  # At least 30 chars
                not response.lower().startswith("i don't know") and
                not response.lower().startswith("i cannot") and
                not response.lower().startswith("sorry")
            )
            
            if is_valid:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
                
            print(f"    {status} - {len(response)} chars")
            if not is_valid:
                print(f"    Response: {response[:150]}")
            
            results.append({
                'question': question,
                'response': response,
                'length': len(response),
                'passed': is_valid
            })
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            failed += 1
            results.append({
                'question': question,
                'error': str(e),
                'passed': False
            })
    
    # Summary
    total = passed + failed
    pass_rate = 100 * passed / total if total > 0 else 0
    
    print(f"\n{'-'*80}")
    print(f"RESULTS:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'-'*80}\n")
    
    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'pass_rate': pass_rate,
        'details': results
    }


def main():
    """Load and test models"""
    
    # Find latest training session
    results_dir = Path("extended_training_results")
    sessions = sorted(results_dir.glob("2*"), reverse=True)
    
    if not sessions:
        print("No training sessions found")
        return
    
    latest_session = sessions[0]
    print(f"\n{'='*80}")
    print(f"MANUAL MODEL EVALUATION")
    print(f"{'='*80}")
    print(f"Session: {latest_session.name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_model_path = latest_session / "base_model.pkl"
    improved_model_path = latest_session / "improved_model.pkl"
    
    all_results = {}
    
    # Test base model
    if base_model_path.exists():
        print(f"\nLoading base model from: {base_model_path}")
        try:
            with open(base_model_path, 'rb') as f:
                base_model = pickle.load(f)
            print("✓ Base model loaded successfully")
            
            base_results = test_model_manually(base_model, "BASE MODEL")
            all_results['base'] = base_results
            
        except Exception as e:
            print(f"✗ Error loading base model: {e}")
            all_results['base'] = {'error': str(e)}
    
    # Test improved model
    if improved_model_path.exists():
        print(f"\nLoading improved model from: {improved_model_path}")
        try:
            with open(improved_model_path, 'rb') as f:
                improved_model = pickle.load(f)
            print("✓ Improved model loaded successfully")
            
            improved_results = test_model_manually(improved_model, "IMPROVED MODEL")
            all_results['improved'] = improved_results
            
        except Exception as e:
            print(f"✗ Error loading improved model: {e}")
            import traceback
            traceback.print_exc()
            all_results['improved'] = {'error': str(e)}
    
    # Comparison
    if 'base' in all_results and 'improved' in all_results:
        if 'error' not in all_results['base'] and 'error' not in all_results['improved']:
            print(f"\n{'='*80}")
            print("COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            base_rate = all_results['base']['pass_rate']
            improved_rate = all_results['improved']['pass_rate']
            improvement = improved_rate - base_rate
            
            print(f"\nBase Model:")
            print(f"  Pass Rate: {base_rate:.1f}%")
            print(f"  Tests: {all_results['base']['passed']}/{all_results['base']['total']}")
            
            print(f"\nImproved Model:")
            print(f"  Pass Rate: {improved_rate:.1f}%")
            print(f"  Tests: {all_results['improved']['passed']}/{all_results['improved']['total']}")
            
            print(f"\nImprovement:")
            if improvement > 0:
                print(f"  ✅ +{improvement:.1f} percentage points")
            elif improvement < 0:
                print(f"  ⚠️  {improvement:.1f} percentage points (regression)")
            else:
                print(f"  → No change")
            
            print(f"{'='*80}\n")
            
            # Save results
            results_file = latest_session / "manual_test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Results saved to: {results_file}\n")
    
    return all_results


if __name__ == "__main__":
    main()
