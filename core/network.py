"""
NeuromorphicNetwork: The colony of BioNodes.
NO layers, NO backprop - just a graph of interconnected neurons.
"""

from .bionode import BioNode


class NeuromorphicNetwork:
    def __init__(self):
        """
        A sparse graph network of neurons.
        Uses dictionaries (adjacency lists) for flash-native scalability.
        """
        self.nodes = {}  # Dict[node_id, BioNode]
        self.current_tick = 0

    def add_node(self, node_id, node_type="hidden", threshold=1.0, refractory_period=3):
        """
        Add a neuron to the network.
        """
        node = BioNode(node_id, node_type, threshold, refractory_period)
        self.nodes[node_id] = node
        return node

    def connect(self, source_id, target_id, weight, is_inhibitory=False):
        """
        Create a synapse from source to target.
        """
        if target_id not in self.nodes:
            raise ValueError(f"Target node {target_id} does not exist")

        self.nodes[target_id].add_input(source_id, weight, is_inhibitory)

    def set_input(self, node_id, value):
        """
        Set external input for an input node.
        """
        if node_id in self.nodes:
            self.nodes[node_id].set_external_input(value)

    def step(self, global_dopamine=0.0, learning_rate=0.01):
        """
        Advance the simulation by one tick.
        All neurons update in parallel (current state affects next state).
        """
        # Capture current network state (which neurons are firing)
        network_state = {
            node_id: node.is_firing
            for node_id, node in self.nodes.items()
        }

        # Update all nodes
        for node in self.nodes.values():
            node.tick(self.current_tick, network_state, global_dopamine, learning_rate)

        self.current_tick += 1

    def get_output(self, node_id):
        """
        Get the spike output of a specific node.
        """
        if node_id in self.nodes:
            return self.nodes[node_id].get_output()
        return 0.0

    def is_firing(self, node_id):
        """
        Check if a node is currently firing.
        """
        return self.nodes.get(node_id, BioNode(-1, "hidden")).is_firing

    def get_weight(self, source_id, target_id):
        """
        Get the current weight of a synapse.
        """
        if target_id in self.nodes:
            if source_id in self.nodes[target_id].inputs:
                return self.nodes[target_id].inputs[source_id].weight
        return 0.0

    def get_node_potential(self, node_id):
        """
        Get the membrane potential of a node.
        """
        if node_id in self.nodes:
            return self.nodes[node_id].potential
        return 0.0

    def reset(self):
        """
        Reset all nodes to resting state (useful for experiments).
        Also clears eligibility traces to prevent cross-contamination between training examples.
        """
        for node in self.nodes.values():
            node.potential = 0.0
            node.is_firing = False
            node.last_spike_tick = -999

            # Clear eligibility traces in all input synapses
            for synapse in node.inputs.values():
                synapse.trace = 0.0

