"""
NCGN v2.0 Linguistic Processor

Peripheral module for translating Natural Language into Graph Triplets.
NO DECISION MAKING. Pure parsing.

Input: "Dogs have fur."
Output: (dog, has_part, fur)
"""

from typing import List, Tuple, Optional, Any, Dict
import json
from .config import Config, DEFAULT_CONFIG
from .schemas.cognitive import KnowledgeGraphUpdate

class LinguisticProcessor:
    def __init__(self, model_path: str, config: Config = DEFAULT_CONFIG):
        self.model_path = model_path
        self.config = config
        self._llm = None
        self._client = None

    def _ensure_loaded(self) -> None:
        """Lazy load the LLM."""
        if self._llm is not None:
            return
        
        try:
            from llama_cpp import Llama
        except ImportError as e:
            print(f"ERROR: Failed to load LLM dependencies: {e}")
            print("Please install: pip install llama-cpp-python")
            return
        
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.config.llm_context_window,
            n_gpu_layers=self.config.llm_gpu_layers,
            chat_format="chatml",
            verbose=True, # Turn on verbose to see C++ logs
        )
        # No instructor patch

    def extract_triplets(self, text: str) -> Optional[KnowledgeGraphUpdate]:
        """
        Parse text into graph structure.
        """
        self._ensure_loaded()
        if not self._llm:
            return None
            
        system_prompt = """You are a linguistic parser for a knowledge graph.
Your ONLY job is to convert English text into structured Nodes and Edges.

JSON Format:
{
  "nodes": [{"id": "node_id", "label": "node_label", "type": "concept"}],
  "edges": [{"source": "source_id", "target": "target_id", "relation": "relation_verb", "confidence": 0.9}]
}
Response MUST be valid JSON only.
"""
        
        user_prompt = f"""Extract knowledge from this text: "{text}"
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            print(f"DEBUG: Sending to LLM raw: {messages}") 
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                response_format={"type": "json_object"} # Force JSON mode if available in new llama-cpp
            )
            print(f"DEBUG: LLM Response raw: {response}")
            
            content = response['choices'][0]['message']['content']
            parsed = json.loads(content)
            
            # --- Robustness Patching for TinyLlama ---
            # 1. Ensure reasoning exists
            if "reasoning" not in parsed:
                parsed["reasoning"] = "Extracted from text (auto-generated)."
            
            # 2. Map loose keys to schema keys
            if "nodes" in parsed and "new_nodes" not in parsed:
                parsed["new_nodes"] = parsed.pop("nodes")
            
            if "edge" in parsed and "new_edges" not in parsed:
                parsed["new_edges"] = parsed.pop("edge")
            elif "edges" in parsed and "new_edges" not in parsed:
                parsed["new_edges"] = parsed.pop("edges")
            elif "edge_ids" in parsed and "new_edges" not in parsed:
                parsed["new_edges"] = parsed.pop("edge_ids")
                
            # 3. Ensure all list fields exist
            for field in ["new_nodes", "new_edges", "nodes_to_remove", "edges_to_weaken"]:
                if field not in parsed:
                    parsed[field] = []
            
            # 4. Strict Sanitization
            valid_nodes = []
            for node in parsed["new_nodes"]:
                # Map 'id' to 'label' if needed
                if "id" in node and "label" not in node:
                    node["label"] = node.pop("id")
                
                # Validation: Label must be string
                if "label" in node and isinstance(node["label"], str):
                    valid_nodes.append(node)
                else:
                    print(f"WARNING: Dropping invalid node: {node}")
            parsed["new_nodes"] = valid_nodes
            
            valid_edges = []
            for edge in parsed["new_edges"]:
                if "relation" in edge and "relation_type" not in edge:
                    edge["relation_type"] = edge.pop("relation")
                    
                # Validation: Source/Target must be strings, Relation Must exist
                if (
                    "source" in edge and isinstance(edge["source"], str) and
                    "target" in edge and isinstance(edge["target"], str) and
                    "relation_type" in edge and isinstance(edge["relation_type"], str)
                ):
                    valid_edges.append(edge)
                else:
                    print(f"WARNING: Skipping invalid edge: {edge}")
            parsed["new_edges"] = valid_edges

            # Convert dict to pydantic model manually if needed, or just return dict if acceptable
            # The codebase expects KnowledgeGraphUpdate object
            return KnowledgeGraphUpdate(**parsed)
            
        except Exception as e:
            print(f"Linguistic processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
