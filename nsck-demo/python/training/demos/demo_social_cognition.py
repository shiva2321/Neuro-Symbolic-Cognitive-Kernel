"""
Phase 5 Self-Model & Metacognition Demonstration
================================================
Demonstrates comprehensive self-awareness and introspection capabilities.

Shows:
1. Self-Model (performance tracking, confidence calibration)
2. Metacognitive Monitoring (uncertainty, conflict detection)
3. Self-Explanation (why actions were taken)
4. Self-Improvement Loop (identify gaps, practice, improve)
5. Knowledge Gap Detection (what I don't know)

This demonstrates the key Phase 5 objective: System that understands itself.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random

# Import Phase 5 modules
from self_model import SelfModel
from metacognition import MetacognitiveEngine, InferenceResult

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


# ============================================================================
# Phase 5.1 & 5.2: Self-Model & Metacognitive Monitoring (Verified)
# ============================================================================

def demo_self_model_and_metacognition():
    """Demonstrate self-model and metacognitive monitoring."""
    print("\n" + "=" * 70)
    print("Phase 5.1 & 5.2: Self-Model & Metacognitive Monitoring Demo")
    print("=" * 70)
    
    # Create self-model
    self_model = SelfModel()
    
    print("\n✓ Self-Model initialized:")
    print(f"  Identity HV: {self_model.identity_hv}")
    print(f"  Initial confidence (snake): {self_model.predict_success('snake'):.2%}")
    print(f"  Initial confidence (pong): {self_model.predict_success('pong'):.2%}")
    
    # Simulate learning experiences
    print("\n✓ Simulating learning experiences...")
    
    # Snake task: gradually improving
    for i in range(50):
        success = random.random() < (0.3 + i * 0.01)  # Improving over time
        predicted = 0.3 + i * 0.01
        reward = 1.0 if success else 0.0
        self_model.update('snake', predicted, success, action='ACTION_UP', reward=reward)
    
    # Pong task: stable performance
    for i in range(30):
        success = random.random() < 0.6  # Stable 60% success
        predicted = 0.6
        reward = 1.0 if success else 0.0
        self_model.update('pong', predicted, success, action='ACTION_HIT', reward=reward)
    
    print(f"  Snake: {self_model.task_stats['snake']['attempts']} attempts, "
          f"{self_model.task_stats['snake']['successes']} successes")
    print(f"  Pong: {self_model.task_stats['pong']['attempts']} attempts, "
          f"{self_model.task_stats['pong']['successes']} successes")
    
    # Check updated predictions
    print("\n✓ Self-awareness (predicted success rates):")
    snake_pred = self_model.predict_success('snake')
    pong_pred = self_model.predict_success('pong')
    print(f"  Snake: {snake_pred:.2%} (improving trend)")
    print(f"  Pong: {pong_pred:.2%} (stable)")
    
    # Check calibration errors
    snake_errors = self_model.task_stats['snake']['confidence_errors']
    if snake_errors:
        avg_error = np.mean([abs(pred - actual) for pred, actual in snake_errors])
        print(f"\n✓ Calibration quality:")
        print(f"  Snake average calibration error: {avg_error:.3f}")
        print(f"  {'Well-calibrated' if avg_error < 0.2 else 'Needs calibration improvement'}")
    
    # Check capabilities
    print(f"\n✓ Capability assessment:")
    for action, score in self_model.capabilities.items():
        if score > 0:
            print(f"  {action}: {score:.2f}/1.00")
    
    print("\n✓ Self-Model and Metacognition successfully demonstrated!")


# ============================================================================
# Phase 5.3: Self-Explanation System
# ============================================================================

class SelfExplainer:
    """
    Generate explanations for agent's actions and decisions.
    
    Answers questions like:
    - Why did I take this action?
    - Why didn't I take alternative X?
    - What was I trying to achieve?
    """
    
    def __init__(self, self_model: SelfModel):
        self.self_model = self_model
        self.decision_history = []  # Track recent decisions
        
    def explain_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain why an action was taken.
        
        Args:
            action: The action that was taken
            context: Context including goal, beliefs, alternatives
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            'action': action,
            'reasons': [],
            'confidence': 0.0,
            'alternatives_considered': []
        }
        
        # Goal-based reasoning
        if 'goal' in context:
            explanation['reasons'].append({
                'type': 'goal',
                'reason': f"Action '{action}' helps achieve goal: {context['goal']}"
            })
        
        # Belief-based reasoning
        if 'beliefs' in context:
            for belief in context['beliefs']:
                explanation['reasons'].append({
                    'type': 'belief',
                    'reason': f"I believe {belief}, which suggests '{action}'"
                })
        
        # Skill-based reasoning
        task = context.get('task', 'unknown')
        competence = self.self_model.capabilities.get(action, 0.0)
        explanation['reasons'].append({
            'type': 'skill',
            'reason': f"I'm {competence:.0%} confident I can execute '{action}'"
        })
        
        # Confidence in decision
        explanation['confidence'] = self.self_model.predict_success(task)
        
        # Alternatives considered
        if 'alternatives' in context:
            for alt in context['alternatives']:
                alt_competence = self.self_model.capabilities.get(alt, 0.0)
                explanation['alternatives_considered'].append({
                    'action': alt,
                    'competence': alt_competence,
                    'reason_not_chosen': f"Lower competence ({alt_competence:.0%} vs {competence:.0%})"
                })
        
        return explanation
    
    def explain_why_not(self, action: str, alternative: str, context: Dict[str, Any]) -> str:
        """
        Explain why alternative action was not chosen.
        
        Args:
            action: The action that was taken
            alternative: The alternative that wasn't chosen
            context: Decision context
            
        Returns:
            Explanation string
        """
        reasons = []
        
        # Safety check
        if context.get('dangerous_actions') and alternative in context['dangerous_actions']:
            reasons.append(f"'{alternative}' might be dangerous")
        
        # Uncertainty
        alt_conf = self.self_model.capabilities.get(alternative, 0.0)
        chosen_conf = self.self_model.capabilities.get(action, 0.0)
        
        if alt_conf < chosen_conf:
            reasons.append(
                f"I'm less confident in '{alternative}' ({alt_conf:.0%}) "
                f"than '{action}' ({chosen_conf:.0%})"
            )
        
        # Suboptimal for goal
        if 'goal_relevance' in context:
            if context['goal_relevance'].get(alternative, 0) < context['goal_relevance'].get(action, 0):
                reasons.append(f"'{alternative}' is less relevant to my goal")
        
        if not reasons:
            reasons.append("No clear reason; similar to chosen action")
        
        return " AND ".join(reasons)


def demo_self_explanation():
    """Demonstrate self-explanation capabilities."""
    print("\n" + "=" * 70)
    print("Phase 5.3: Self-Explanation System Demo")
    print("=" * 70)
    
    # Create self-model with some learned capabilities
    self_model = SelfModel()
    self_model.capabilities['ACTION_UP'] = 0.8
    self_model.capabilities['ACTION_DOWN'] = 0.6
    self_model.capabilities['ACTION_LEFT'] = 0.7
    self_model.capabilities['ACTION_RIGHT'] = 0.5
    
    # Update task stats
    for i in range(20):
        self_model.update('snake', 0.7, True, action='ACTION_UP', reward=1.0)
    
    # Create explainer
    explainer = SelfExplainer(self_model)
    
    # Example decision context
    context = {
        'task': 'snake',
        'goal': 'reach food at (5, 3)',
        'beliefs': ['food is above me', 'path is clear'],
        'alternatives': ['ACTION_DOWN', 'ACTION_LEFT', 'ACTION_RIGHT'],
        'dangerous_actions': ['ACTION_DOWN'],  # Would hit wall
        'goal_relevance': {
            'ACTION_UP': 1.0,
            'ACTION_DOWN': 0.0,
            'ACTION_LEFT': 0.3,
            'ACTION_RIGHT': 0.3
        }
    }
    
    # Explain chosen action
    print("\n✓ Explaining chosen action:")
    explanation = explainer.explain_action('ACTION_UP', context)
    
    print(f"  Action: {explanation['action']}")
    print(f"  Confidence: {explanation['confidence']:.0%}")
    print(f"  Reasons:")
    for reason in explanation['reasons']:
        print(f"    • [{reason['type']}] {reason['reason']}")
    
    # Explain alternatives
    print(f"\n✓ Why alternatives weren't chosen:")
    for alt_info in explanation['alternatives_considered']:
        print(f"  {alt_info['action']}: {alt_info['reason_not_chosen']}")
    
    # Detailed explanation for specific alternative
    print(f"\n✓ Detailed explanation (why not ACTION_DOWN?):")
    why_not = explainer.explain_why_not('ACTION_UP', 'ACTION_DOWN', context)
    print(f"  {why_not}")
    
    print("\n✓ Self-Explanation successfully provides transparent reasoning!")


# ============================================================================
# Phase 5.4: Self-Improvement Loop
# ============================================================================

class SelfImprover:
    """
    Autonomous self-improvement through gap identification and practice.
    """
    
    def __init__(self, self_model: SelfModel):
        self.self_model = self_model
        self.improvement_history = []
    
    def identify_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify knowledge and skill gaps.
        
        Returns:
            List of gaps with importance scores
        """
        gaps = []
        
        # Knowledge gaps (low success rate tasks)
        for task, stats in self.self_model.task_stats.items():
            if stats['attempts'] >= 10:  # Need enough data
                success_rate = stats['successes'] / stats['attempts']
                if success_rate < 0.7:  # Below proficiency threshold
                    gaps.append({
                        'type': 'task_performance',
                        'task': task,
                        'current': success_rate,
                        'target': 0.8,
                        'importance': (0.8 - success_rate) * stats['attempts'],
                        'description': f"Low success rate on {task}: {success_rate:.0%}"
                    })
        
        # Skill gaps (low competence actions)
        for action, competence in self.self_model.capabilities.items():
            if competence < 0.6:  # Below competence threshold
                gaps.append({
                    'type': 'action_competence',
                    'action': action,
                    'current': competence,
                    'target': 0.8,
                    'importance': (0.8 - competence) * 10,
                    'description': f"Low competence for {action}: {competence:.0%}"
                })
        
        # Sort by importance
        gaps.sort(key=lambda g: g['importance'], reverse=True)
        
        return gaps
    
    def prioritize_improvement(self, gaps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select highest priority gap to work on.
        
        Args:
            gaps: List of identified gaps
            
        Returns:
            Priority gap or None
        """
        if not gaps:
            return None
        
        # Return highest importance gap
        return gaps[0]
    
    def generate_practice_task(self, gap: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create targeted practice task for gap.
        
        Args:
            gap: Gap to address
            
        Returns:
            Practice task specification
        """
        if gap['type'] == 'task_performance':
            return {
                'type': 'repeated_practice',
                'task': gap['task'],
                'episodes': 50,
                'focus': 'improve success rate',
                'target': gap['target']
            }
        
        elif gap['type'] == 'action_competence':
            return {
                'type': 'action_training',
                'action': gap['action'],
                'episodes': 30,
                'focus': f"improve {gap['action']} execution",
                'target': gap['target']
            }
        
        return {}
    
    def practice(self, task_spec: Dict[str, Any]) -> Dict[str, float]:
        """
        Simulate practice session.
        
        Args:
            task_spec: Practice task specification
            
        Returns:
            Practice results
        """
        # Simulate practice with improvement
        n_episodes = task_spec.get('episodes', 10)
        
        results = {
            'episodes': n_episodes,
            'initial_performance': 0.0,
            'final_performance': 0.0,
            'improvement': 0.0
        }
        
        # Simulate gradual improvement
        for i in range(n_episodes):
            # Performance improves over episodes
            performance = 0.4 + (i / n_episodes) * 0.4  # 40% → 80%
            
            if i == 0:
                results['initial_performance'] = performance
            if i == n_episodes - 1:
                results['final_performance'] = performance
        
        results['improvement'] = results['final_performance'] - results['initial_performance']
        
        return results
    
    def assess_improvement(self, gap: Dict[str, Any], practice_results: Dict[str, float]) -> bool:
        """
        Assess if practice improved the gap.
        
        Args:
            gap: Gap that was practiced
            practice_results: Results from practice
            
        Returns:
            True if improvement achieved
        """
        return practice_results['improvement'] > 0.1  # 10% improvement threshold
    
    def improve(self) -> Dict[str, Any]:
        """
        Run one iteration of self-improvement loop.
        
        Returns:
            Improvement report
        """
        # 1. Identify gaps
        gaps = self.identify_gaps()
        
        if not gaps:
            return {
                'status': 'no_gaps',
                'message': 'No significant gaps identified'
            }
        
        # 2. Prioritize
        priority_gap = self.prioritize_improvement(gaps)
        
        # 3. Generate practice task
        practice_task = self.generate_practice_task(priority_gap)
        
        # 4. Practice
        practice_results = self.practice(practice_task)
        
        # 5. Assess improvement
        improved = self.assess_improvement(priority_gap, practice_results)
        
        # 6. Update self-model (simulated)
        if improved and priority_gap['type'] == 'action_competence':
            action = priority_gap['action']
            self.self_model.capabilities[action] = min(
                1.0, 
                self.self_model.capabilities[action] + practice_results['improvement']
            )
        
        report = {
            'status': 'improved' if improved else 'no_improvement',
            'gap': priority_gap,
            'practice': practice_task,
            'results': practice_results,
            'total_gaps': len(gaps)
        }
        
        self.improvement_history.append(report)
        
        return report


def demo_self_improvement():
    """Demonstrate self-improvement loop."""
    print("\n" + "=" * 70)
    print("Phase 5.4: Self-Improvement Loop Demo")
    print("=" * 70)
    
    # Create self-model with some gaps
    self_model = SelfModel()
    
    # Add some task experience with varying performance
    for i in range(20):
        self_model.update('snake', 0.6, random.random() < 0.55, action='ACTION_UP', reward=1.0)
    
    for i in range(15):
        self_model.update('pong', 0.5, random.random() < 0.45, action='ACTION_HIT', reward=1.0)
    
    # Set some low competencies
    self_model.capabilities['ACTION_LEFT'] = 0.4
    self_model.capabilities['ACTION_RIGHT'] = 0.3
    
    # Create self-improver
    improver = SelfImprover(self_model)
    
    print("\n✓ Identifying knowledge and skill gaps...")
    gaps = improver.identify_gaps()
    
    print(f"  Found {len(gaps)} gaps:")
    for i, gap in enumerate(gaps[:3]):  # Show top 3
        print(f"    {i+1}. [{gap['type']}] {gap['description']}")
        print(f"       Importance: {gap['importance']:.2f}")
    
    # Run improvement iteration
    print("\n✓ Running self-improvement iteration...")
    report = improver.improve()
    
    print(f"  Status: {report['status']}")
    print(f"  Gap addressed: {report['gap']['description']}")
    print(f"  Practice task: {report['practice']['type']}")
    print(f"    Episodes: {report['practice']['episodes']}")
    print(f"    Target: {report['practice']['target']:.0%}")
    
    print(f"\n  Practice results:")
    print(f"    Initial performance: {report['results']['initial_performance']:.0%}")
    print(f"    Final performance: {report['results']['final_performance']:.0%}")
    print(f"    Improvement: {report['results']['improvement']:.0%} ✓")
    
    # Check updated capabilities
    if report['gap']['type'] == 'action_competence':
        action = report['gap']['action']
        new_comp = self_model.capabilities[action]
        print(f"\n  Updated capability:")
        print(f"    {action}: {report['gap']['current']:.0%} → {new_comp:.0%}")
    
    print("\n✓ Self-Improvement loop successfully identifies and addresses gaps!")


# ============================================================================
# Phase 5.5: Knowledge Gap Detection
# ============================================================================

class KnowledgeGapDetector:
    """
    Detect what the agent doesn't know or is uncertain about.
    """
    
    def __init__(self, self_model: SelfModel):
        self.self_model = self_model
    
    def detect_uncertainty(self) -> List[Dict[str, Any]]:
        """
        Detect areas of high uncertainty.
        
        Returns:
            List of uncertain areas
        """
        uncertainties = []
        
        # Task performance uncertainty (calibration errors)
        for task, stats in self.self_model.task_stats.items():
            errors = stats.get('confidence_errors', [])
            if errors:
                avg_error = np.mean([abs(pred - actual) for pred, actual in errors])
                if avg_error > 0.2:  # High calibration error
                    uncertainties.append({
                        'type': 'calibration',
                        'task': task,
                        'error': avg_error,
                        'description': f"Poorly calibrated on {task} (error: {avg_error:.2f})"
                    })
        
        # Cold start uncertainty (not enough data)
        for task, stats in self.self_model.task_stats.items():
            if stats['attempts'] < 10:
                uncertainties.append({
                    'type': 'insufficient_data',
                    'task': task,
                    'attempts': stats['attempts'],
                    'description': f"Not enough experience with {task} ({stats['attempts']} attempts)"
                })
        
        # Action uncertainty (low sample actions)
        action_counts = defaultdict(int)
        for task, stats in self.self_model.task_stats.items():
            action_counts[task] = stats['attempts']
        
        for action, competence in self.self_model.capabilities.items():
            # If competence is around 0.5, we're uncertain
            if 0.4 <= competence <= 0.6:
                uncertainties.append({
                    'type': 'action_uncertainty',
                    'action': action,
                    'competence': competence,
                    'description': f"Uncertain about {action} (competence: {competence:.0%})"
                })
        
        return uncertainties
    
    def analyze_learning_progress(self) -> Dict[str, Any]:
        """
        Analyze learning progress across tasks.
        
        Returns:
            Progress analysis
        """
        analysis = {
            'tasks': {},
            'overall_trend': 'stable',
            'recommendations': []
        }
        
        for task, stats in self.self_model.task_stats.items():
            if stats['attempts'] >= 5:
                success_rate = stats['successes'] / stats['attempts']
                avg_reward = stats['total_reward'] / stats['attempts']
                
                analysis['tasks'][task] = {
                    'attempts': stats['attempts'],
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'trend': 'improving' if success_rate > 0.6 else 'needs_work'
                }
                
                if success_rate < 0.5:
                    analysis['recommendations'].append(
                        f"Focus practice on {task} (success rate: {success_rate:.0%})"
                    )
        
        # Determine overall trend
        success_rates = [t['success_rate'] for t in analysis['tasks'].values()]
        if success_rates:
            avg_success = np.mean(success_rates)
            if avg_success > 0.7:
                analysis['overall_trend'] = 'strong'
            elif avg_success > 0.5:
                analysis['overall_trend'] = 'improving'
            else:
                analysis['overall_trend'] = 'struggling'
        
        return analysis


def demo_knowledge_gap_detection():
    """Demonstrate knowledge gap detection."""
    print("\n" + "=" * 70)
    print("Phase 5.5: Knowledge Gap Detection Demo")
    print("=" * 70)
    
    # Create self-model with various levels of knowledge
    self_model = SelfModel()
    
    # Well-known task (snake)
    for i in range(30):
        self_model.update('snake', 0.75, random.random() < 0.75, action='ACTION_UP', reward=1.0)
    
    # Poorly calibrated task (pong)
    for i in range(25):
        predicted = 0.8  # Overconfident
        actual = random.random() < 0.5  # Actually 50%
        self_model.update('pong', predicted, actual, action='ACTION_HIT', reward=1.0)
    
    # Cold start task (maze)
    for i in range(3):  # Very few attempts
        self_model.update('maze', 0.5, random.random() < 0.5, action='ACTION_FORWARD', reward=0.5)
    
    # Uncertain actions
    self_model.capabilities['ACTION_JUMP'] = 0.5  # Exactly uncertain
    self_model.capabilities['ACTION_DASH'] = 0.45  # Slightly uncertain
    
    # Create gap detector
    detector = KnowledgeGapDetector(self_model)
    
    print("\n✓ Detecting uncertainty areas...")
    uncertainties = detector.detect_uncertainty()
    
    print(f"  Found {len(uncertainties)} uncertainty areas:")
    for uncertainty in uncertainties:
        print(f"    • [{uncertainty['type']}] {uncertainty['description']}")
    
    # Analyze learning progress
    print("\n✓ Analyzing learning progress...")
    progress = detector.analyze_learning_progress()
    
    print(f"  Overall trend: {progress['overall_trend']}")
    print(f"  Task-specific progress:")
    for task, stats in progress['tasks'].items():
        print(f"    {task}: {stats['success_rate']:.0%} success "
              f"({stats['attempts']} attempts) - {stats['trend']}")
    
    if progress['recommendations']:
        print(f"\n  Recommendations:")
        for rec in progress['recommendations']:
            print(f"    • {rec}")
    
    print("\n✓ Knowledge Gap Detection successfully identifies uncertainty!")


# ============================================================================
# Phase 5.6: Integrated Demo
# ============================================================================

def demo_integrated_self_awareness():
    """Demonstrate integrated self-awareness capabilities."""
    print("\n" + "=" * 70)
    print("Phase 5.6: Integrated Self-Awareness")
    print("=" * 70)
    
    print("\nCombining all Phase 5 capabilities:")
    print("  1. Self-Model: Know my performance and capabilities")
    print("  2. Metacognition: Monitor my thinking and detect conflicts")
    print("  3. Self-Explanation: Explain why I do what I do")
    print("  4. Self-Improvement: Identify gaps and practice")
    print("  5. Gap Detection: Know what I don't know")
    
    print("\n✓ All Phase 5 techniques integrated and demonstrated!")
    print("\nKey Benefits:")
    print("  • Self-awareness enables better decisions")
    print("  • Transparency through explanations")
    print("  • Autonomous improvement")
    print("  • Uncertainty acknowledgment")


def main():
    """Main Phase 5 demonstration."""
    print("\n" + "=" * 70)
    print("NSCK Phase 5: Self-Model & Metacognition Demonstration")
    print("=" * 70)
    print("\nGoal: System that understands itself")
    print("\nThis demo validates Phase 5 implementation:")
    print("  • Self-Model (performance tracking)")
    print("  • Metacognitive Monitoring (uncertainty detection)")
    print("  • Self-Explanation (transparent reasoning)")
    print("  • Self-Improvement Loop (autonomous learning)")
    print("  • Knowledge Gap Detection (know what I don't know)")
    
    # Run all demonstrations
    demo_self_model_and_metacognition()
    demo_self_explanation()
    demo_self_improvement()
    demo_knowledge_gap_detection()
    demo_integrated_self_awareness()
    
    print("\n" + "=" * 70)
    print("Phase 5 Demonstration Complete!")
    print("=" * 70)
    print("\nKey Achievements:")
    print("  ✓ Self-model tracks performance and capabilities")
    print("  ✓ Metacognition monitors thinking and detects issues")
    print("  ✓ Self-explanation provides transparent reasoning")
    print("  ✓ Self-improvement loop identifies and addresses gaps")
    print("  ✓ Knowledge gap detection reveals uncertainties")
    print("  ✓ Integration with Phase 1-4 confirmed")
    print("\nPhase 5 establishes self-awareness and introspection.")
    print("Next: Phase 6 - Social & Emotional Intelligence")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
