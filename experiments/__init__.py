"""
Neuromorphic Learning Experiments
Demonstrations of the NCGN system learning various tasks.

Experiments:
- Pavlov: Classical conditioning with dopamine-modulated STDP
- Sequence: Temporal pattern learning and prediction
- XOR: Non-linear classification with lateral inhibition
- UnifiedLearner: Multi-task learning in a single network
"""

def __getattr__(name):
    """Lazy loading of experiment classes"""
    if name == 'PavlovExperiment':
        from .pavlov_experiment import PavlovExperiment
        return PavlovExperiment
    elif name == 'SequenceLearningExperiment':
        from .sequence_experiment import SequenceLearningExperiment
        return SequenceLearningExperiment
    elif name == 'XORExperiment':
        from .xor_experiment import XORExperiment
        return XORExperiment
    elif name == 'UnifiedLearner':
        from .unified_learner import UnifiedLearner
        return UnifiedLearner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'PavlovExperiment',
    'SequenceLearningExperiment',
    'XORExperiment',
    'UnifiedLearner'
]

