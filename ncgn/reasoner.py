"""
NCGN v7 LLM Reasoner Module

System 2: Local LLM with structured output.

Uses instructor + llama-cpp-python for schema-enforced generation.
Implements reflexion loop for validation and Chain-of-Thought prompting.

Features:
- Pydantic schema enforcement
- Automatic retry on validation failure
- Chain-of-Thought reasoning (reasoning field generated first)
- Context injection from active concepts
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .topology import GraphTopology
    from .state import CognitiveState

from .config import Config, DEFAULT_CONFIG
from .schemas.cognitive import (
    KnowledgeGraphUpdate,
    QueryResponse,
    DiagnosisResult,
    ConceptNode,
    RelationEdge,
)


class LLMReasoner:
    """
    System 2: Local LLM with structured output.
    
    Uses instructor + llama-cpp-python for schema-enforced generation.
    The LLM cannot directly output actions - it must propose graph
    updates that are then applied by the Brain.
    
    Features:
    - Pydantic schema enforcement via instructor
    - Automatic retry on validation failure (reflexion)
    - Chain-of-Thought: reasoning field generated first
    - Context injection from active concepts
    """
    
    def __init__(
        self,
        model_path: str,
        config: Config = DEFAULT_CONFIG
    ):
        """
        Initialize the LLM reasoner.
        
        Args:
            model_path: Path to GGUF model file
            config: Configuration object
        """
        self.model_path = model_path
        self.config = config
        
        # Lazy load model
        self._llm = None
        self._client = None
        
        # Statistics
        self._total_calls = 0
        self._total_retries = 0
        self._total_failures = 0
    
    def _ensure_loaded(self) -> None:
        """Lazy load the LLM and instructor client."""
        if self._llm is not None:
            return
        
        try:
            from llama_cpp import Llama
            import instructor
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python and instructor are required for LLMReasoner. "
                "Install with: pip install llama-cpp-python instructor"
            ) from e
        
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.config.llm_context_window,
            n_gpu_layers=self.config.llm_gpu_layers,
            chat_format="chatml",
            verbose=False,
        )
        
        # Patch with instructor for structured output
        self._client = instructor.patch(
            create=self._llm.create_chat_completion,
            mode=instructor.Mode.JSON_SCHEMA
        )
    
    def _format_context(self, active_concepts: Dict[str, float]) -> str:
        """Format active concepts as context string."""
        if not active_concepts:
            return "(no active concepts)"
        
        # Sort by activation, take top K
        sorted_concepts = sorted(
            active_concepts.items(),
            key=lambda x: -x[1]
        )[:self.config.top_k_concepts]
        
        return ", ".join([
            f"{label} ({energy:.2f})"
            for label, energy in sorted_concepts
        ])
    
    def extract_knowledge(
        self,
        user_input: str,
        active_concepts: Dict[str, float],
        subgraph_description: str = ""
    ) -> Optional[KnowledgeGraphUpdate]:
        """
        Extract knowledge graph updates from user input.
        
        Args:
            user_input: The user's text input
            active_concepts: Currently active concepts and their energies
            subgraph_description: Text description of relevant subgraph
            
        Returns:
            KnowledgeGraphUpdate with new nodes/edges, or None on failure
        """
        self._ensure_loaded()
        self._total_calls += 1
        
        context = self._format_context(active_concepts)
        
        system_prompt = """You are the Logic Core of NCGN, a cognitive graph network.
Your job is to organize chaotic associations into coherent knowledge.

RULES:
1. Always explain your reasoning FIRST before proposing changes
2. Use lowercase labels with underscores (e.g., "red_apple" not "Red Apple")
3. Only add nodes/edges that are clearly implied by the input
4. Relation types should be simple verbs or "is_a", "has_property", etc.
5. Weights between 0.3-0.9 (0.5 = typical, 0.8+ = strong)"""

        user_prompt = f"""Active concepts: {context}

{f"Graph context: {subgraph_description}" if subgraph_description else ""}

User input: "{user_input}"

Analyze this input and determine what knowledge to add or update."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            result = self._client.chat.completions.create(
                messages=messages,
                response_model=KnowledgeGraphUpdate,
                max_retries=self.config.llm_max_retries,
            )
            return result
        except Exception as e:
            self._total_failures += 1
            print(f"LLM extraction failed: {e}")
            return None
    
    def answer_query(
        self,
        question: str,
        active_concepts: Dict[str, float],
        subgraph_description: str = ""
    ) -> Optional[QueryResponse]:
        """
        Answer a user question using graph context.
        
        Args:
            question: The user's question
            active_concepts: Currently active concepts
            subgraph_description: Text description of relevant subgraph
            
        Returns:
            QueryResponse with answer and confidence, or None on failure
        """
        self._ensure_loaded()
        self._total_calls += 1
        
        context_labels = ", ".join(list(active_concepts.keys())[:30])
        
        system_prompt = """You are answering questions based on your knowledge graph.
Be honest about uncertainty. Think step-by-step before answering.

RULES:
1. Explain your reasoning FIRST
2. Base your answer on the known concepts
3. Set confidence lower if unsure (0.3-0.5 for uncertain)
4. List any missing knowledge that would help"""

        user_prompt = f"""Known concepts: {context_labels}

{f"Graph context: {subgraph_description}" if subgraph_description else ""}

Question: {question}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            result = self._client.chat.completions.create(
                messages=messages,
                response_model=QueryResponse,
                max_retries=self.config.llm_max_retries,
            )
            return result
        except Exception as e:
            self._total_failures += 1
            print(f"LLM query failed: {e}")
            return None
    
    def diagnose_anomaly(
        self,
        triple: tuple,  # (agent, action, object)
        active_concepts: Dict[str, float],
        known_properties: Dict[str, Dict[str, bool]]
    ) -> Optional[DiagnosisResult]:
        """
        Diagnose why an action seems anomalous.
        
        Args:
            triple: (agent, action, object) tuple
            active_concepts: Currently active concepts
            known_properties: Properties database
            
        Returns:
            DiagnosisResult with diagnosis type and suggested action
        """
        self._ensure_loaded()
        self._total_calls += 1
        
        agent, action, obj = triple
        
        # Format known properties
        props_str = ""
        if agent in known_properties:
            props_str += f"{agent} properties: {known_properties[agent]}\n"
        if obj in known_properties:
            props_str += f"{obj} properties: {known_properties[obj]}\n"
        
        system_prompt = """You are diagnosing anomalies in a cognitive graph.

Diagnosis types:
- constraint_violation: The action violates known constraints
- missing_knowledge: We don't know enough to evaluate
- no_issue: The action seems fine

Suggested actions:
- apply_ltd: Weaken the association that led to this
- query_user: Ask the user for clarification
- no_action: No intervention needed"""

        user_prompt = f"""Action to evaluate: {agent} --{action}--> {obj}

{props_str if props_str else "No known properties"}

Active context: {', '.join(list(active_concepts.keys())[:20])}

Is there any issue with this action?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            result = self._client.chat.completions.create(
                messages=messages,
                response_model=DiagnosisResult,
                max_retries=self.config.llm_max_retries,
            )
            return result
        except Exception as e:
            self._total_failures += 1
            print(f"LLM diagnosis failed: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._llm is not None
    
    def get_stats(self) -> Dict:
        """Get reasoner statistics."""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded(),
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "success_rate": (
                (self._total_calls - self._total_failures) / max(self._total_calls, 1)
            ),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"LLMReasoner(loaded={stats['is_loaded']}, "
            f"calls={stats['total_calls']})"
        )


class MockReasoner:
    """
    Mock reasoner for testing without LLM.
    
    Returns empty updates, allowing System 1 to run standalone.
    """
    
    def __init__(self):
        self._total_calls = 0
    
    def extract_knowledge(
        self,
        user_input: str,
        active_concepts: Dict[str, float],
        subgraph_description: str = ""
    ) -> KnowledgeGraphUpdate:
        """Return empty update."""
        self._total_calls += 1
        return KnowledgeGraphUpdate(
            reasoning="Mock reasoner - no actual LLM processing",
            new_nodes=[],
            new_edges=[],
        )
    
    def answer_query(
        self,
        question: str,
        active_concepts: Dict[str, float],
        subgraph_description: str = ""
    ) -> QueryResponse:
        """Return placeholder answer."""
        self._total_calls += 1
        return QueryResponse(
            reasoning="Mock reasoner - using active concepts without LLM",
            answer=f"Active concepts: {', '.join(list(active_concepts.keys())[:10])}",
            confidence=0.1,
        )
    
    def is_loaded(self) -> bool:
        return True
    
    def get_stats(self) -> Dict:
        return {
            "type": "mock",
            "total_calls": self._total_calls,
        }
