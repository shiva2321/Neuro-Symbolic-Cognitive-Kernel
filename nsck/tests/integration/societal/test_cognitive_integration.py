import pytest
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.integration.config import NSCKConfig

def test_cognitive_engine_societal_integration():
    config = NSCKConfig()
    config.enable_active_inference = True
    config.enable_dual_process = False # Force system 2 to use narrative
    
    engine = CognitiveEngine(config=config)
    engine.register_task("test_world")
    
    # 1. Ingest concepts so the world isn't empty
    world = engine.semantic_memory.societal_world
    
    # Needs to be dictionary state for default adapter
    state = {"text": "hello", "position_x": 0, "position_y": 1}
    
    # Mock some data
    world.ingest_concept("start_pos", engine.get_concept_hv("start_pos"))
    world.ingest_concept("end_pos", engine.get_concept_hv("end_pos"))
    world.tick_world()
    
    # 2. Run Decide
    decision = engine.decide(state, "test_world", fast_mode=False)
    
    # Assert returning a valid CognitiveState
    assert decision.task_tag == "test_world"
    
    # We should have generated a Narrative coalition that competed in GWT.
    # It might not win, but the fact that `decide` runs without crashing means
    # NarrativeEngine and SocietalRouter are properly linked up.
    
    assert engine.narrative_engine is not None
    assert engine.societal_router is not None
    assert engine.semantic_memory.societal_world is not None
    
    # Test Active Inference Free Energy social stability bias
    fe = engine.active_inference.free_energy("start_pos", engine.get_concept_hv("start_pos"))
    assert isinstance(fe, float)

def test_schema_snapshot_store():
    config = NSCKConfig()
    engine = CognitiveEngine(config=config)
    
    # Try capturing a snapshot
    from python.core.memory.schema_snapshot_store import SchemaSnapshotStore
    store = SchemaSnapshotStore(max_snapshots=5)
    
    schema = {"name": "test", "properties": {"color": "red"}}
    store.capture_snapshot("test", schema)
    
    history = store.get_history("test")
    assert len(history) == 1
    assert history[0]["schema"]["properties"]["color"] == "red"
