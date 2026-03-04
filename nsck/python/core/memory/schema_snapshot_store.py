import time
from typing import Dict, List, Any

class SchemaSnapshotStore:
    """
    Maintains historical snapshots of conceptual schemas to handle polysemy 
    and track schema evolution over the semantic space.
    """
    
    def __init__(self, max_snapshots: int = 50):
        self.snapshots: Dict[str, List[Dict[str, Any]]] = {}
        self.max_snapshots = max_snapshots
        
    def capture_snapshot(self, concept_id: str, schema_dict: Dict[str, Any]):
        """Capture the current state of a concept's schema."""
        if concept_id not in self.snapshots:
            self.snapshots[concept_id] = []
            
        snapshot = {
            "epoch": int(time.time()),
            "schema": dict(schema_dict) # Shallow copy is usually fine for these dicts
        }
        
        history = self.snapshots[concept_id]
        history.append(snapshot)
        
        # Enforce rolling history
        if len(history) > self.max_snapshots:
            history.pop(0)
            
    def get_latest_snapshot(self, concept_id: str) -> Dict[str, Any]:
        """Return the most recent schema snapshot."""
        if concept_id not in self.snapshots or not self.snapshots[concept_id]:
            return {}
        return self.snapshots[concept_id][-1]["schema"]
        
    def get_history(self, concept_id: str) -> List[Dict[str, Any]]:
        """Return the full history of a given concept's schema."""
        return self.snapshots.get(concept_id, [])
