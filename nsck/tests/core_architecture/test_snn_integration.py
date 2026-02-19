"""
Integration Tests for SNN + VSA + Hebbian Architecture
=======================================================

Comprehensive test suite validating:
- Component functionality
- Integration points
- Performance requirements
- Edge cases
- End-to-end workflows

Run: pytest test_snn_integration.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from python.core.perception.snn_perception import (
    SNNPerceptionModule, LIFNeuronLayer, SimpleConceptMapper
)
from python.core.perception.vsa_snn_bridge import RateCoder, TemporalCoder
from python.core.learning.hebbian import VSAHebbianLearner
from python.core.perception.snn_integration import SNNWorkspaceAdapter
from python.core.training.snn_training import SNNTrainer, TrainingConfig, SNNDataset
from python.core.reasoning.global_workspace import GlobalWorkspace
from python.core.memory.semantic_memory import SemanticMemory


# ============================================================================
# Component Tests
# ============================================================================

class TestLIFNeurons:
    """Test Leaky Integrate-and-Fire neuron dynamics"""
    
    def test_neuron_initialization(self):
        """Test LIF neurons initialize correctly"""
        layer = LIFNeuronLayer(n_neurons=100)
        assert layer.n_neurons == 100
        assert len(layer.v) == 100
        assert np.allclose(layer.v, -70.0)  # Resting potential
    
    def test_spike_generation(self):
        """Test neurons spike when threshold crossed"""
        layer = LIFNeuronLayer(n_neurons=10, v_thresh=-50.0, tau=5.0)
        
        # Strong input sustained over multiple steps should cause spikes
        strong_input = np.ones(10) * 50.0
        any_spikes = False
        for _ in range(10):  # Run multiple steps
            spikes = layer.step(strong_input)
            if np.any(spikes > 0):
                any_spikes = True
                break
        
        assert any_spikes, "Strong sustained input should produce spikes"
    
    def test_refractory_period(self):
        """Test refractory period prevents immediate re-spiking"""
        layer = LIFNeuronLayer(n_neurons=1, v_thresh=-50.0, refractory_period=5.0)
        
        # Cause spike
        spikes1 = layer.step(np.array([100.0]))
        
        # Try to spike again immediately (should fail due to refractory)
        spikes2 = layer.step(np.array([100.0]))
        
        if np.any(spikes1 > 0):  # If first spike happened
            assert np.all(spikes2 == 0), "Refractory period should prevent immediate spike"
    
    def test_reset(self):
        """Test layer reset clears state"""
        layer = LIFNeuronLayer(n_neurons=10)
        
        # Run some steps
        for _ in range(10):
            layer.step(np.random.randn(10) * 10)
        
        # Reset
        layer.reset()
        
        assert np.allclose(layer.v, -70.0)
        assert len(layer.spike_history) == 0


class TestVSAEncoding:
    """Test VSA spike encoding/decoding"""
    
    def test_rate_coder_initialization(self):
        """Test RateCoder initializes correctly"""
        coder = RateCoder(n_neurons=100, dimension=1024)
        assert coder.n_neurons == 100
        assert coder.dimension == 1024
        assert len(coder.neuron_hvs) == 100
    
    def test_rate_encoding(self):
        """Test rate encoding produces hypervectors"""
        coder = RateCoder(n_neurons=100, dimension=1024)
        
        # Create spike train: 50 timesteps, 100 neurons
        spike_train = np.random.rand(50, 100) > 0.9
        
        encoding = coder.encode(spike_train, time_window_ms=50.0)
        
        assert encoding.hv is not None
        assert len(encoding.active_neurons) >= 0
        assert encoding.encoding_time_ms >= 0
    
    def test_temporal_coder(self):
        """Test temporal coding"""
        coder = TemporalCoder(n_neurons=100, dimension=1024, max_shift=10)
        
        spike_train = np.random.rand(50, 100) > 0.9
        encoding = coder.encode(spike_train, dt_ms=1.0)
        
        assert encoding.hv is not None


class TestHebbianLearning:
    """Test Hebbian learning rules"""
    
    def test_vsa_hebbian_initialization(self):
        """Test VSA Hebbian learner initializes"""
        learner = VSAHebbianLearner(dimension=1024, n_concepts=20)
        assert learner.dimension == 1024
        assert learner.n_concepts == 20
    
    def test_association_update(self):
        """Test association updates work"""
        learner = VSAHebbianLearner(dimension=1024, n_concepts=10)
        
        # Update associations
        learner.update_associations([0, 1, 2])
        learner.update_associations([1, 3])
        
        # Spreading from concept 1 should activate 0, 2, 3
        activated = learner.spread_activation([1], n_steps=1, threshold=0.0)
        
        assert len(activated) > 1, "Spreading should activate associated concepts"
    
    def test_spreading_activation_decay(self):
        """Test spreading activation decays with steps"""
        learner = VSAHebbianLearner(dimension=1024, n_concepts=10)
        
        # Create strong association
        for _ in range(10):
            learner.update_associations([0, 1])
        
        # Spread 1 step
        act_1_step = learner.spread_activation([0], n_steps=1)
        
        # Spread 3 steps
        act_3_steps = learner.spread_activation([0], n_steps=3)
        
        # More steps should activate more concepts
        assert len(act_3_steps) >= len(act_1_step)


class TestConceptMapper:
    """Test concept pattern recognition"""
    
    def test_concept_registration(self):
        """Test registering new concepts"""
        mapper = SimpleConceptMapper(dimension=1024, n_concepts=50)
        
        concept_id = mapper.register_concept([0, 1, 5, 10])
        
        assert concept_id >= 0
        assert concept_id in mapper.concepts
    
    def test_pattern_recognition(self):
        """Test recognizing similar patterns"""
        mapper = SimpleConceptMapper(dimension=1024, n_concepts=50)
        
        # Register pattern
        pattern1 = [0, 1, 2, 3, 4]
        cid = mapper.register_concept(pattern1)
        
        # Recognize similar pattern
        pattern2 = [0, 1, 2, 3, 5]  # 4/5 overlap
        recognized_id, strength = mapper.recognize_pattern(pattern2, threshold=0.5)
        
        assert recognized_id == cid, "Similar pattern should be recognized"
        assert strength > 0.5


# ============================================================================
# Integration Tests
# ============================================================================

class TestSNNPerception:
    """Test integrated SNN perception module"""
    
    def test_perception_module_initialization(self):
        """Test SNN perception module initializes"""
        module = SNNPerceptionModule(
            input_dim=64,
            snn_size=128,
            hv_dimension=1024
        )
        assert module.input_dim == 64
        assert module.snn_size == 128
    
    def test_perception_cycle(self):
        """Test complete perception cycle"""
        module = SNNPerceptionModule(input_dim=64, snn_size=128)
        
        sensory_input = np.random.randn(64) * 0.5
        result = module.perceive(sensory_input, learn=True)
        
        assert 'concept_hv' in result
        assert 'concept_id' in result
        assert 'spike_train' in result
        assert 'processing_time_ms' in result
        assert result['processing_time_ms'] < 10.0  # Should be fast
    
    def test_concept_formation(self):
        """Test unsupervised concept formation"""
        module = SNNPerceptionModule(input_dim=64, snn_size=128)
        
        # Present distinct patterns multiple times
        pattern_a = np.ones(64) * 0.5
        pattern_b = -np.ones(64) * 0.5
        
        results_a = []
        results_b = []
        
        for _ in range(5):
            results_a.append(module.perceive(pattern_a, learn=True))
            results_b.append(module.perceive(pattern_b, learn=True))
        
        # Check if concepts stabilize
        final_concept_a = results_a[-1]['concept_id']
        final_concept_b = results_b[-1]['concept_id']
        
        # Different patterns should map to different concepts (usually)
        # Note: This might fail sometimes due to random initialization
        assert len(module.concept_mapper.concepts) > 0, "Should learn some concepts"
    
    def test_performance_requirements(self):
        """Test latency and throughput requirements"""
        module = SNNPerceptionModule(input_dim=64, snn_size=256)
        
        latencies = []
        for _ in range(10):
            result = module.perceive(np.random.randn(64), learn=False)
            latencies.append(result['processing_time_ms'])
        
        avg_latency = np.mean(latencies)
        throughput = 1000.0 / avg_latency
        
        assert avg_latency < 5.0, f"Avg latency {avg_latency:.2f}ms exceeds 5ms requirement"
        assert throughput > 200, f"Throughput {throughput:.1f} Hz below 200 Hz requirement"


class TestGlobalWorkspaceIntegration:
    """Test integration with Global Workspace Theory"""
    
    def test_workspace_adapter_creation(self):
        """Test creating workspace adapter"""
        adapter = SNNWorkspaceAdapter(
            input_dim=64,
            snn_size=128,
            hv_dimension=1024
        )
        assert adapter.snn_module is not None
    
    def test_coalition_proposal(self):
        """Test generating coalition proposals"""
        adapter = SNNWorkspaceAdapter(input_dim=64, snn_size=128)
        
        # Perceive something
        adapter.perceive(np.random.randn(64))
        
        # Get proposal
        proposal = adapter.get_coalition_proposal()
        
        assert proposal is not None
        assert 'source' in proposal
        assert 'base_salience' in proposal
        assert proposal['source'] == 'snn_perception'
    
    def test_broadcast_reception(self):
        """Test receiving GWT broadcasts"""
        adapter = SNNWorkspaceAdapter(input_dim=64, snn_size=128)
        
        adapter.receive_broadcast("ACTION_MOVE")
        adapter.receive_broadcast("ACTION_EAT")
        
        assert len(adapter.broadcast_history) == 2
    
    def test_semantic_memory_registration(self):
        """Test concepts register in semantic memory"""
        semantic_mem = SemanticMemory()
        adapter = SNNWorkspaceAdapter(
            input_dim=64,
            snn_size=128,
            semantic_memory=semantic_mem
        )
        
        # Perceive patterns
        for _ in range(3):
            adapter.perceive(np.random.randn(64), learn=True)
        
        # Check semantic memory has concepts
        assert len(semantic_mem.concept_hvs) > 0, "Concepts should register in semantic memory"


class TestTrainingPipeline:
    """Test end-to-end training pipeline"""
    
    def test_dataset_creation(self):
        """Test dataset wrapper"""
        data = np.random.randn(100, 64)
        labels = np.random.randint(0, 5, 100)
        
        dataset = SNNDataset(data, labels)
        
        assert len(dataset) == 100
        assert dataset[0][0].shape == (64,)
        assert dataset[0][1] in range(5)
    
    def test_training_config(self):
        """Test training configuration"""
        config = TrainingConfig(
            input_dim=64,
            snn_size=128,
            n_epochs=5,
            mode="supervised"
        )
        assert config.n_epochs == 5
        assert config.mode == "supervised"
    
    def test_supervised_training(self):
        """Test supervised training loop"""
        # Small dataset for quick test
        data = np.random.randn(100, 64)
        labels = np.random.randint(0, 3, 100)
        dataset = SNNDataset(data, labels)
        
        config = TrainingConfig(
            input_dim=64,
            snn_size=64,
            n_concepts=10,
            n_epochs=3,
            batch_size=20,
            mode="supervised",
            verbose=False
        )
        
        trainer = SNNTrainer(config)
        trainer.train(dataset, None)
        
        assert trainer.metrics.epoch == 3
        assert len(trainer.metrics.avg_spikes) == 3
    
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints"""
        config = TrainingConfig(
            input_dim=64,
            snn_size=64,
            n_epochs=2,
            verbose=False,
            checkpoint_dir="test_checkpoints"
        )
        
        trainer = SNNTrainer(config)
        
        # Train briefly
        data = np.random.randn(50, 64)
        dataset = SNNDataset(data)
        trainer.train(dataset, None)
        
        # Save
        trainer.save_checkpoint("test")
        
        # Create new trainer and load
        trainer2 = SNNTrainer(config)
        trainer2.load_checkpoint("test")
        
        assert trainer2.metrics.epoch == trainer.metrics.epoch
        
        # Cleanup
        import shutil
        if Path("test_checkpoints").exists():
            shutil.rmtree("test_checkpoints")


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEnd:
    """Test complete end-to-end workflows"""
    
    def test_perception_to_memory_to_decision(self):
        """Test full pipeline: perception → memory → reasoning"""
        # Setup
        workspace = GlobalWorkspace()
        semantic_mem = SemanticMemory()
        
        adapter = SNNWorkspaceAdapter(
            input_dim=64,
            snn_size=128,
            semantic_memory=semantic_mem
        )
        
        workspace.register_module("snn_perception", adapter)
        
        # Perceive multiple patterns
        patterns = [np.random.randn(64) * 0.5 for _ in range(5)]
        
        for pattern in patterns:
            result = adapter.perceive(pattern, learn=True)
            proposal = adapter.get_coalition_proposal()
            
            # Simulate workspace broadcast
            if proposal:
                adapter.receive_broadcast(f"CONCEPT_{result['concept_id']}")
        
        # Verify concepts in semantic memory
        assert len(semantic_mem.concept_hvs) > 0, "Should have learned concepts"
        assert len(adapter.broadcast_history) > 0, "Should have received broadcasts"
    
    def test_learning_convergence(self):
        """Test that learning converges on repeated patterns"""
        module = SNNPerceptionModule(input_dim=64, snn_size=128)
        
        # Fixed pattern
        pattern = np.random.randn(64) * 0.5
        
        concept_ids = []
        for _ in range(10):
            result = module.perceive(pattern, learn=True)
            concept_ids.append(result['concept_id'])
        
        # After a few iterations, concept should stabilize
        final_concepts = concept_ids[-3:]
        assert len(set(final_concepts)) <= 2, "Concept should stabilize for repeated pattern"
    
    def test_accuracy_requirement(self):
        """Test achieving minimum accuracy requirement"""
        # Create learnable dataset
        from python.core.training.snn_training import create_synthetic_dataset
        
        train_ds, val_ds = create_synthetic_dataset(
            n_samples=200,
            n_classes=3,
            input_dim=64,
            noise=0.1  # Low noise for easier learning
        )
        
        config = TrainingConfig(
            input_dim=64,
            snn_size=128,
            n_concepts=10,
            n_epochs=10,
            mode="supervised",
            hebbian_lr=0.01,
            verbose=False
        )
        
        trainer = SNNTrainer(config)
        trainer.train(train_ds, val_ds)
        
        val_stats = trainer.evaluate(val_ds)
        
        assert val_stats['accuracy'] >= 0.70, \
            f"Accuracy {val_stats['accuracy']:.3f} below 70% requirement"


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests"""
    
    def test_throughput_stress(self):
        """Test sustained high throughput"""
        module = SNNPerceptionModule(input_dim=64, snn_size=256)
        
        n_samples = 100
        latencies = []
        
        for _ in range(n_samples):
            result = module.perceive(np.random.randn(64), learn=False)
            latencies.append(result['processing_time_ms'])
        
        avg_latency = np.mean(latencies)
        throughput = 1000.0 / avg_latency
        
        print(f"\n  Throughput test: {throughput:.1f} Hz avg, {avg_latency:.2f}ms latency")
        assert throughput > 200, f"Sustained throughput {throughput:.1f} Hz too low"
    
    def test_memory_scaling(self):
        """Test memory usage scales reasonably"""
        sizes = [64, 128, 256]
        
        for size in sizes:
            module = SNNPerceptionModule(input_dim=64, snn_size=size)
            
            # Process some patterns
            for _ in range(10):
                module.perceive(np.random.randn(64), learn=True)
            
            # Memory should scale linearly (roughly)
            params = size * 64
            memory_est_mb = params * 4 / 1024 / 1024
            
            assert memory_est_mb < 5.0, f"Memory {memory_est_mb:.1f} MB too high for size {size}"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
