"""Quick debug script to test STDP learning."""
from core.plasticity import STDPRule
from core.synapse import Synapse

stdp = STDPRule(learning_rate=0.01)
syn = Synapse(weight=0.5)
syn.trace = 1.0

print(f'Before: {syn.weight}')
delta = stdp.apply(syn, post_fired=True, dopamine=1.0)
print(f'After: {syn.weight}, delta={delta}')
print(f'Weight increased: {syn.weight > 0.5}')
